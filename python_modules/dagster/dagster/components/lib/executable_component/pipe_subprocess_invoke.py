import os
import shutil
from typing import TYPE_CHECKING, Any, Union

from dagster._core.execution.context.asset_check_execution_context import AssetCheckExecutionContext
from dagster._core.execution.context.asset_execution_context import AssetExecutionContext
from dagster._core.pipes.subprocess import PipesSubprocessClient
from dagster.components.core.context import ComponentLoadContext

if TYPE_CHECKING:
    from dagster.components.lib.executable_component.component import ExecutableComponent


def invoke_pipes_subprocess_script(
    component: "ExecutableComponent",
    context: Union[AssetExecutionContext, AssetCheckExecutionContext],
    path: str,
    component_load_context: ComponentLoadContext,
) -> Any:
    assert not component.resource_keys, "Pipes subprocess scripts cannot have resources"
    path = path if os.path.isabs(path) else os.path.join(component_load_context.path, path)
    cmd = [shutil.which("python"), path]
    return (
        PipesSubprocessClient().run(context=context.op_execution_context, command=cmd).get_results()
    )
