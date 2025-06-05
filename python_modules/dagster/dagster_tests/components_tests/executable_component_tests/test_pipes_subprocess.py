import inspect
from textwrap import dedent

from dagster._core.definitions.materialize import materialize
from dagster._core.definitions.metadata.metadata_value import TextMetadataValue
from dagster.components.lib.executable_component.component import PipesSubprocessSpec
from dagster.components.lib.executable_component.subprocess_component import SubprocessComponent
from dagster.components.testing import scaffold_defs_sandbox


def test_pipes_subprocess_script_hello_world() -> None:
    with scaffold_defs_sandbox(component_cls=SubprocessComponent) as sandbox:
        execute_path = sandbox.defs_folder_path / "script.py"
        execute_path.write_text("print('hello world')")

        with sandbox.load(
            component_body={
                "type": "dagster.components.lib.executable_component.subprocess_component.SubprocessComponent",
                "attributes": {
                    "execution": {
                        "type": "subprocess",
                        "name": "op_name",
                        "path": "script.py",
                    },
                    "assets": [
                        {
                            "key": "asset",
                        }
                    ],
                },
            }
        ) as (component, defs):
            assert isinstance(component, SubprocessComponent)
            assert isinstance(component.execution, PipesSubprocessSpec)
            assets_def = defs.get_assets_def("asset")
            result = materialize([assets_def])
            assert result.success
            mats = result.asset_materializations_for_node("op_name")
            assert len(mats) == 1


def test_pipes_subprocess_script_with_custom_materialize_result() -> None:
    def code_to_copy():
        from dagster_pipes import open_dagster_pipes

        if __name__ == "__main__":
            with open_dagster_pipes() as context:
                context.report_asset_materialization(metadata={"foo": "bar"})

    with scaffold_defs_sandbox(component_cls=SubprocessComponent) as sandbox:
        raw_source = inspect.getsource(code_to_copy)
        raw_source = "\n".join(raw_source.split("\n")[1:])
        raw_source = dedent(raw_source)

        execute_path = sandbox.defs_folder_path / "op_name.py"
        execute_path.write_text(raw_source)

        with sandbox.load(
            component_body={
                "type": "dagster.components.lib.executable_component.subprocess_component.SubprocessComponent",
                "attributes": {
                    "execution": {
                        "type": "subprocess",
                        "path": "op_name.py",
                    },
                    "assets": [
                        {
                            "key": "asset",
                        }
                    ],
                },
            }
        ) as (component, defs):
            assert isinstance(component, SubprocessComponent)
            assert isinstance(component.execution, PipesSubprocessSpec)
            assets_def = defs.get_assets_def("asset")
            result = materialize([assets_def])
            assert result.success
            mats = result.asset_materializations_for_node("op_name")
            assert len(mats) == 1
            assert mats[0].metadata == {"foo": TextMetadataValue("bar")}
