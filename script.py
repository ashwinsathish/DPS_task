from azureml.core import Environment
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice

myenv = Environment.from_conda_specification(name="myenv", file_path="./myenv.yml")

inference_config = InferenceConfig(entry_script="score.py", environment=myenv)

deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

service = Model.deploy(workspace=ws,
                       name="my-service",
                       models=[model],
                       inference_config=inference_config,
                       deployment_config=deployment_config,
                       overwrite=True)

service.wait_for_deployment(show_output=True)
