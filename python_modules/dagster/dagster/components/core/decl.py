import abc
import inspect
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Generic, Optional, TypeVar, Union

from dagster_shared.serdes.objects import PluginObjectKey
from dagster_shared.yaml_utils import parse_yamls_with_source_position
from dagster_shared.yaml_utils.source_position import SourcePosition, ValueAndSourcePositionTree
from pydantic import TypeAdapter

import dagster._check as check
from dagster._core.errors import DagsterInvalidDefinitionError
from dagster._utils.pydantic_yaml import (
    _parse_and_populate_model_with_annotated_errors,
    enrich_validation_errors_with_source_position,
)
from dagster.components.component.component import Component
from dagster.components.component.component_loader import is_component_loader
from dagster.components.core.context import ComponentLoadContext
from dagster.components.core.defs_module import (
    EXPLICITLY_IGNORED_GLOB_PATTERNS,
    ComponentFileModel,
    ComponentPath,
    CompositeComponent,
    CompositeYamlComponent,
    DagsterDefsComponent,
    DefsFolderComponent,
    context_with_injected_scope,
    find_defs_or_component_yaml,
)
from dagster.components.core.package_entry import load_package_object

T = TypeVar("T", bound=Component)


class ComponentDecl(abc.ABC, Generic[T]):
    """Class representing a not-yet-loaded component in the defs hierarchy. A tree of component decls
    is initially built before any components are loaded.
    """

    @abc.abstractmethod
    def _load_component(self) -> T: ...


class ComponentDeclWithChildren(ComponentDecl[T]):
    def iterate_component_decls(self) -> Iterator[ComponentDecl]:
        for _, component in self.iterate_path_component_decl_pairs():
            yield component

    @abc.abstractmethod
    def iterate_path_component_decl_pairs(
        self,
    ) -> Iterator[tuple[ComponentPath, ComponentDecl]]: ...


class ComponentLoaderDecl(ComponentDecl[Component]):
    def __init__(
        self,
        context: ComponentLoadContext,
        path: ComponentPath,
        component_node_fn: Callable[[ComponentLoadContext], Component],
    ):
        self.context = context
        self.path = path
        self.component_node = component_node_fn

    def _load_component(self) -> Component:
        return self.component_node(self.context)


class CompositePythonDecl(ComponentDeclWithChildren[CompositeComponent]):
    def __init__(self, context: ComponentLoadContext, decls: Mapping[str, ComponentLoaderDecl]):
        self.context = context
        self.decls = decls

    def _load_component(self) -> "CompositeComponent":
        return CompositeComponent(
            components={
                attr: self.context.component_tree.load_component_at_path(decl.path)
                for attr, decl in self.decls.items()
            }
        )

    def iterate_path_component_decl_pairs(
        self,
    ) -> Iterator[tuple[ComponentPath, ComponentDecl]]:
        for decl in self.decls.values():
            yield decl.path, decl


class YamlDecl(ComponentDecl):
    @staticmethod
    def from_source_tree(
        context: ComponentLoadContext,
        source_tree: ValueAndSourcePositionTree,
        path: ComponentPath,
    ) -> "YamlDecl":
        component_file_model = _parse_and_populate_model_with_annotated_errors(
            cls=ComponentFileModel, obj_parse_root=source_tree, obj_key_path_prefix=[]
        )
        return YamlDecl(
            context=context,
            source_tree=source_tree,
            component_file_model=component_file_model,
            path=path,
        )

    def __init__(
        self,
        context: ComponentLoadContext,
        source_tree: ValueAndSourcePositionTree,
        component_file_model: ComponentFileModel,
        path: ComponentPath,
    ):
        self.context = context
        self.source_tree = source_tree
        self.component_file_model = component_file_model
        self.path = path

    def _load_component(self) -> "Component":
        # find the component type
        type_str = self.context.normalize_component_type_str(self.component_file_model.type)
        key = PluginObjectKey.from_typename(type_str)
        obj = load_package_object(key)
        if not isinstance(obj, type) or not issubclass(obj, Component):
            raise DagsterInvalidDefinitionError(
                f"Component type {type_str} is of type {type(obj)}, but must be a subclass of dagster.Component"
            )

        context = context_with_injected_scope(
            self.context, obj, self.component_file_model.template_vars_module
        )

        context = context.with_source_position_tree(
            self.source_tree.source_position_tree,
        )

        model_cls = obj.get_model_cls()

        # grab the attributes from the yaml file
        if model_cls is None:
            attributes = None
        elif self.source_tree:
            attributes_position_tree = self.source_tree.source_position_tree.children["attributes"]
            with enrich_validation_errors_with_source_position(
                attributes_position_tree, ["attributes"]
            ):
                attributes = TypeAdapter(model_cls).validate_python(
                    self.component_file_model.attributes
                )
        else:
            attributes = TypeAdapter(model_cls).validate_python(
                self.component_file_model.attributes
            )

        return obj.load(attributes, context)


class CompositeYamlDecl(ComponentDeclWithChildren[CompositeYamlComponent]):
    def __init__(
        self,
        context: ComponentLoadContext,
        decls: Sequence[YamlDecl],
        source_positions: Sequence[SourcePosition],
    ):
        self.context = context
        self.decls = decls
        self.source_positions = source_positions

    def _load_component(self) -> "CompositeYamlComponent":
        return CompositeYamlComponent(
            components=[
                self.context.component_tree.load_component_at_path(decl.path) for decl in self.decls
            ],
            source_positions=self.source_positions,
        )

    def iterate_path_component_decl_pairs(
        self,
    ) -> Iterator[tuple[ComponentPath, ComponentDecl]]:
        for decl in self.decls:
            yield decl.path, decl


class DagsterDefsDecl(ComponentDecl[DagsterDefsComponent]):
    def __init__(self, context: ComponentLoadContext, path: Path):
        self.path = path

    def _load_component(self) -> DagsterDefsComponent:
        return DagsterDefsComponent(path=self.path)


class DefsFolderDecl(ComponentDeclWithChildren[DefsFolderComponent]):
    def __init__(
        self, context: ComponentLoadContext, path: Path, children: Mapping[Path, ComponentDecl]
    ):
        self.path = path
        self.children = children

    @classmethod
    def get(cls, context: ComponentLoadContext) -> "DefsFolderDecl":
        component = get_component_decl(context)
        return check.inst(
            component,
            DefsFolderDecl,
            f"Expected PendingDefsFolderComponent at {context.path}, got {component}.",
        )

    def _load_component(self) -> "DefsFolderComponent":
        return DefsFolderComponent(
            path=self.path,
            children={subpath: decl._load_component() for subpath, decl in self.children.items()},  # noqa: SLF001
            asset_post_processors=None,
        )

    def iterate_path_component_decl_pairs(
        self,
    ) -> Iterator[tuple[ComponentPath, ComponentDecl]]:
        for path, component_node in self.children.items():
            yield ComponentPath(file_path=path), component_node

            if isinstance(component_node, ComponentDeclWithChildren):
                yield from component_node.iterate_path_component_decl_pairs()


def get_component_decl(context: ComponentLoadContext) -> Optional[ComponentDecl]:
    """Attempts to determine the type of component that should be loaded for the given context.  Iterates through potential component
    type matches, prioritizing more specific types: YAML, Python, plain Dagster defs, and component
    folder.
    """
    # in priority order
    # yaml component
    if find_defs_or_component_yaml(context.path):
        return load_yaml_component_decl(context)
    # pythonic component
    elif (
        context.terminate_autoloading_on_keyword_files and (context.path / "component.py").exists()
    ):
        return get_component_decl_from_python_file(context)
    # defs
    elif (
        context.terminate_autoloading_on_keyword_files
        and (context.path / "definitions.py").exists()
    ):
        return DagsterDefsDecl(context=context, path=context.path / "definitions.py")
    elif context.path.suffix == ".py":
        return DagsterDefsDecl(context=context, path=context.path)
    # folder
    elif context.path.is_dir():
        children = find_component_decls_from_context(context)
        if children:
            return DefsFolderDecl(
                context=context,
                path=context.path,
                children=children,
            )

    return None


def find_component_decls_from_context(
    context: ComponentLoadContext,
) -> Mapping[Path, ComponentDecl]:
    found = {}
    for subpath in context.path.iterdir():
        relative_subpath = subpath.relative_to(context.path)
        if any(relative_subpath.match(pattern) for pattern in EXPLICITLY_IGNORED_GLOB_PATTERNS):
            continue
        component_node = get_component_decl(context.for_path(subpath))
        if component_node:
            found[subpath] = component_node
    return found


def get_component_decl_from_python_file(
    context: ComponentLoadContext,
) -> Union[ComponentLoaderDecl, CompositePythonDecl]:
    # backcompat for component.yaml
    component_def_path = context.path / "component.py"
    module = context.load_defs_relative_python_module(component_def_path)
    component_loaders = list(inspect.getmembers(module, is_component_loader))
    if len(component_loaders) == 0:
        raise DagsterInvalidDefinitionError("No component nodes found in module")
    elif len(component_loaders) == 1:
        _, component_loader = component_loaders[0]
        return ComponentLoaderDecl(
            context=context,
            component_node_fn=component_loader,
            path=ComponentPath(file_path=context.path, instance_key=None),
        )
    else:
        return CompositePythonDecl(
            context=context,
            decls={
                attr: ComponentLoaderDecl(
                    context=context,
                    component_node_fn=component_loader,
                    path=ComponentPath(file_path=context.path, instance_key=attr),
                )
                for attr, component_loader in component_loaders
            },
        )


def load_yaml_component_decl(context: ComponentLoadContext) -> ComponentDecl:
    component_def_path = check.not_none(find_defs_or_component_yaml(context.path))
    return get_component_decl_from_yaml_file(context=context, component_def_path=component_def_path)


def get_component_decl_from_yaml_file(
    context: ComponentLoadContext, component_def_path: Path
) -> ComponentDecl:
    source_trees = parse_yamls_with_source_position(
        component_def_path.read_text(), str(component_def_path)
    )
    component_nodes = []
    for i, source_tree in enumerate(source_trees):
        print(context.path, i)
        component_nodes.append(
            YamlDecl.from_source_tree(
                context=context,
                source_tree=source_tree,
                path=ComponentPath(file_path=context.path, instance_key=i),
            )
        )

    check.invariant(len(component_nodes) > 0, "No components found in YAML file")
    return CompositeYamlDecl(
        context=context,
        decls=component_nodes,
        source_positions=[
            source_tree.source_position_tree.position for source_tree in source_trees
        ],
    )
