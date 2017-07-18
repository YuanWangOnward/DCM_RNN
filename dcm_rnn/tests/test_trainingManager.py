from unittest import TestCase
import training_manager
from unittest import TestCase
import toolboxes as tb
import tf_model as tfm

class TestTrainingManager(TestCase):
    def setUp(self):
        self.tm = training_manager.TrainingManager()
        self.tm.N_PACKAGES = 1
        self.PROJECT_DIR = tb.setup_module()
        # load in SPM_data
        data_path = self.PROJECT_DIR + "/dcm_rnn/resources/template0.pkl"
        self.du = tb.load_template(data_path)
        self.dr = tfm.DcmRnn()
        self.dr.collect_parameters(self.du)
        self.tm.prepare_dcm_rnn(self.dr, tag='initializer')
        # dr.build_an_initializer_graph()

    def tearDown(self):
        del self.tm
        del self.du
        del self.dr

    def test_modify_signel_data_package(self):
        tm = self.tm

        configure_package = self.tm.prepare_distributed_configure_package()
        configure_package = configure_package[0]
        data_package = self.tm.prepare_data(self.du, self.dr, configure_package)
        data_package = self.tm.modify_signel_data_package(data_package, 'du', self.du)
        self.assertTrue('du' in data_package.data.keys(), 'modify_signel_data_package error')




