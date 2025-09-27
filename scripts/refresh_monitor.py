import argparse
from databricks.connect import DatabricksSession
from databricks.sdk import WorkspaceClient

from mlchapter_project01.config import ProjectConfig
from mlchapter_project01.monitoring import create_or_refresh_monitoring

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--env",
    action="store",
    default=None,
    type=str,
    required=True,
)

args = parser.parse_args()
root_path = args.root_path
config_path = f"{root_path}/files/project_config_mlchapter.yml"

# Load configuration
config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)

spark = DatabricksSession.builder.getOrCreate()
workspace = WorkspaceClient()

create_or_refresh_monitoring(config=config, spark=spark, workspace=workspace)
