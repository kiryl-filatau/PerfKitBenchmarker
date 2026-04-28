import unittest

from absl.testing import flagsaver
from absl.testing import parameterized
import mock
from perfkitbenchmarker import errors
from perfkitbenchmarker.resources.container_service import kubectl
from perfkitbenchmarker.resources.container_service import kubernetes_cluster
from perfkitbenchmarker.resources.container_service import kubernetes_commands
from perfkitbenchmarker.resources.kubernetes import wg_serving_inference_server
from tests import pkb_common_test_case

_BENCHMARK_SPEC_YAML = """
cluster_boot:
  container_cluster:
    cloud: GCP
    type: Autopilot
    vm_count: 1
    vm_spec: *default_dual_core
    inference_server:
      model_server: vllm
      hf_token: gs://bucket/path/to/token
      model_name: llama3-8b
      catalog_components: 1-L4
      hpa_max_replicas: 10
      extra_deployment_args:
        container-image: vllm/vllm-openai:v0.8.5
"""

_INFERENCE_SERVER_MANIFEST = """
kind: Service
metadata:
  name: test-service
spec:
  selector:
    app: test-app
  ports:
  - port: 80
---
kind: Deployment
metadata:
  name: test-deployment
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: inference-server
        env:
        - name: MODEL_ID
          value: test-model
"""


class WgServingInferenceServerTest(pkb_common_test_case.PkbCommonTestCase):

  def setUp(self):
    super().setUp()
    self.mock_cluster = self.enter_context(
        mock.patch.object(
            kubernetes_cluster, 'KubernetesCluster', autospec=True
        )
    )
    self.mock_run_kubectl = self.enter_context(
        mock.patch.object(kubectl, 'RunKubectlCommand', autospec=True)
    )
    self.config_spec = pkb_common_test_case.CreateBenchmarkSpecFromYaml(
        _BENCHMARK_SPEC_YAML
    )
    self.server = wg_serving_inference_server.WGServingInferenceServer(
        spec=self.config_spec.config.container_cluster.inference_server,
        cluster=self.mock_cluster,
    )

  @parameterized.parameters(
      dict(
          catalog_components='v6e-2x2',
          expected_accelerator_type='v6e',
          expected_accelerator_count=4,
      ),
      dict(
          catalog_components='1-L4',
          expected_accelerator_type='L4',
          expected_accelerator_count=1,
      ),
      dict(
          catalog_components='gcsfuse,8-H100',
          expected_accelerator_type='H100',
          expected_accelerator_count=8,
      ),
      dict(
          catalog_components='gcsfuse',
          expected_accelerator_type='unknown',
          expected_accelerator_count=0,
      ),
  )
  def testMetadataAcceleratorType(
      self,
      catalog_components,
      expected_accelerator_type,
      expected_accelerator_count,
  ):
    modified_spec = _BENCHMARK_SPEC_YAML.replace(
        'catalog_components: 1-L4', f'catalog_components: {catalog_components}'
    )
    self.config_spec = pkb_common_test_case.CreateBenchmarkSpecFromYaml(
        modified_spec
    )
    self.server = wg_serving_inference_server.WGServingInferenceServer(
        spec=self.config_spec.config.container_cluster.inference_server,
        cluster=self.mock_cluster,
    )
    metadata = self.server.GetResourceMetadata()
    self.assertEqual(metadata['accelerator_type'], expected_accelerator_type)
    self.assertEqual(metadata['accelerator_count'], expected_accelerator_count)

  @parameterized.named_parameters(
      dict(
          testcase_name='gcp_gcsfuse',
          cloud='GCP',
          catalog_components='gcsfuse,8-H100',
          expected_storage='gcsfuse',
      ),
      dict(
          testcase_name='azure_blobfuse2',
          cloud='Azure',
          catalog_components='blobfuse2,1-L4',
          expected_storage='blobfuse2',
      ),
      dict(
          testcase_name='azure_gcsfuse_fallback',
          cloud='Azure',
          catalog_components='gcsfuse',
          expected_storage='huggingface',
      ),
      dict(
          testcase_name='gcp_blobfuse2_fallback',
          cloud='GCP',
          catalog_components='blobfuse2',
          expected_storage='huggingface',
      ),
      dict(
          testcase_name='default_hf',
          cloud='GCP',
          catalog_components='1-L4',
          expected_storage='huggingface',
      ),
  )
  def testGetStorageType(self, cloud, catalog_components, expected_storage):
    with flagsaver.flagsaver(cloud=cloud):
      self.server.spec.catalog_components = catalog_components
      self.assertEqual(self.server.GetStorageType(), expected_storage)

  def testValidateFuse_raises_gcsfuse_when_not_gcp(self):
    with flagsaver.flagsaver(cloud='Azure'):
      self.server.spec.catalog_components = 'gcsfuse'
      with self.assertRaises(errors.Resource.CreationError):
        self.server._ValidateFuseCatalogVersusCloud()

  def testValidateFuse_raises_blobfuse2_when_not_azure(self):
    with flagsaver.flagsaver(cloud='GCP'):
      self.server.spec.catalog_components = 'blobfuse2'
      with self.assertRaises(errors.Resource.CreationError):
        self.server._ValidateFuseCatalogVersusCloud()

  def testValidateFuse_raises_both_tokens(self):
    with flagsaver.flagsaver(cloud='GCP'):
      self.server.spec.catalog_components = 'gcsfuse,blobfuse2'
      with self.assertRaises(errors.Resource.CreationError):
        self.server._ValidateFuseCatalogVersusCloud()

  @mock.patch.object(
      wg_serving_inference_server.kubernetes_commands,
      'ApplyManifest',
      return_value=[],
  )
  @mock.patch(
      'perfkitbenchmarker.providers.azure.util.GetAzureStorageAccountKey',
      return_value='fake-account-key',
  )
  def testApplyAzureBlobFusePVC_applies_manifest(
      self,
      unused_key_mock,
      apply_manifest_mock,
  ):
    self.server.created_resources = []
    with flagsaver.flagsaver(
        cloud='Azure',
        k8s_inference_server_azure_blob_storage_account='myacct',
        k8s_inference_server_azure_blob_container='mycontainer',
        k8s_inference_server_azure_blob_resource_group='myrg',
        k8s_inference_server_azure_blob_secret_name='my-secret',
    ):
      self.server._ApplyAzureBlobFusePVC()
    apply_manifest_mock.assert_called_once()
    self.assertEqual(
        apply_manifest_mock.call_args.args[0],
        'container/kubernetes_ai_inference/azurefuse_pv_pvc.yaml.j2',
    )
    kwargs = apply_manifest_mock.call_args.kwargs
    self.assertEqual(
        kwargs['secret_name'],
        'my-secret',
    )
    self.assertEqual(kwargs['volume_handle'], 'myacct_mycontainer')
    self.assertEqual(kwargs['storage_account'], 'myacct')
    self.assertEqual(kwargs['container_name'], 'mycontainer')
    self.assertEqual(kwargs['resource_group'], 'myrg')

  def testApplyAzureBlobFusePVC_requires_account_and_container(self):
    with flagsaver.flagsaver(cloud='Azure'):
      with self.assertRaises(errors.Resource.CreationError):
        self.server._ApplyAzureBlobFusePVC()

  def testDelete(self):
    self.mock_run_kubectl.return_value = ('', '', 0)
    # No assertions, but it runs without error.
    self.server.Delete()

  def testParseAndStoreInferenceServerDetails(self):
    self.server.deployment_metadata = None
    self.server._ParseAndStoreInferenceServerDetails(_INFERENCE_SERVER_MANIFEST)
    self.assertEqual(self.server.service_name, 'test-service')
    self.assertEqual(self.server.service_port, 80)
    self.assertEqual(self.server.app_selector, 'test-app')
    self.assertEqual(self.server.model_id, 'test-model')
    self.assertIsNotNone(self.server.deployment_metadata)

  @mock.patch.object(
      kubernetes_commands,
      'ApplyManifest',
      return_value=['job/test-job'],
  )
  @mock.patch.object(
      kubernetes_commands,
      'RetryableGetPodNameFromJob',
      return_value='test-pod',
  )
  @mock.patch.object(
      kubernetes_commands,
      'GetFileContentFromPod',
      return_value=(_INFERENCE_SERVER_MANIFEST),
  )
  @mock.patch.object(kubernetes_commands, 'DeleteResource')
  def testGetInferenceServerManifest(
      self,
      delete_resource_mock,
      get_file_content_from_pod_mock,
      retryable_get_pod_name_from_job_mock,
      apply_manifest_mock,
  ):
    self.server.spec.model_server = 'vllm'
    self.server.spec.model_name = 'model1'
    self.server.spec.cloud = 'gcp'
    self.server.spec.catalog_components = 'gcsfuse'
    self.server.spec.extra_deployment_args = {}
    self.server.spec.runtime_class_name = 'test-runtime'
    manifest = self.server._GetInferenceServerManifest()
    self.assertIn('runtimeClassName: test-runtime', manifest)
    self.assertIn('kind: Deployment', manifest)
    delete_resource_mock.assert_called_with(
        'job/test-job', ignore_not_found=True
    )

  @parameterized.parameters(
      dict(
          node_labels={
              'node.kubernetes.io/instance-type': 'g2-standard-8',
              'nvidia.com/gpu.product': 'L4',
          },
          expected_metadata={
              'node_name': 'test-node',
              'node_machine_type': 'g2-standard-8',
              'node_machine_family': 'g2',
              'gpu': 'L4',
          },
          description='GCP',
      ),
      dict(
          node_labels={
              'node.kubernetes.io/instance-type': 'g6.xlarge',
              'karpenter.k8s.aws/instance-family': 'g6',
              'nvidia.com/gpu.product': 'L4',
          },
          expected_metadata={
              'node_name': 'test-node',
              'node_machine_type': 'g6.xlarge',
              'node_machine_family': 'g6',
              'gpu': 'L4',
          },
          description='AWS',
      ),
      dict(
          node_labels={
              'beta.kubernetes.io/instance-type': 'Standard_NC6s_v3',
              'nvidia.com/gpu.product': 'T4',
          },
          expected_metadata={
              'node_name': 'test-node',
              'node_machine_type': 'Standard_NC6s_v3',
              'node_machine_family': 'Standard',
              'gpu': 'T4',
          },
          description='Azure',
      ),
  )
  @mock.patch.object(kubernetes_commands, 'GetResourceMetadataByName')
  def testMonitorPodStartupNodeMetadata(
      self, mock_grmbn, node_labels, expected_metadata, description
  ):
    """Tests node metadata collection in _MonitorPodStartup across clouds."""
    pod_name = 'test-pod'
    timestamp = '2024-01-01T00:00:00Z'

    pod_metadata = {
        'metadata': {'creationTimestamp': timestamp},
        'spec': {'nodeName': 'test-node'},
        'status': {
            'conditions': [
                {'type': 'PodScheduled', 'lastTransitionTime': timestamp},
                {'type': 'Ready', 'lastTransitionTime': timestamp},
            ],
            'containerStatuses': [{
                'name': 'inference-server',
                'state': {'running': {'startedAt': timestamp}},
            }],
        },
    }

    node_metadata = {'metadata': {'labels': node_labels}}

    mock_grmbn.return_value = node_metadata
    self.server.GetInferenceServerLogsFromPod = mock.Mock(return_value='logs')
    self.server.GetPodTimeZone = mock.Mock(return_value='UTC')
    self.server.timezone = None

    result = self.server._MonitorPodStartup(pod_name, pod_metadata)

    self.assertIsNotNone(result)
    for key, value in expected_metadata.items():
      self.assertEqual(
          result.metadata.get(key), value, f'{description}: {key} mismatch'
      )


if __name__ == '__main__':
  unittest.main()
