import logging
import grpc

from fedml.core.distributed.communication.constants import CommunicationConstants


#class FedMLCommManager(Observer):

def _init_manager(self):
    logging.info("DQIHGFUQWFJHFBQJHFQBWKHF")

    if self.backend == "MPI":
        from fedml.core.distributed.communication.mpi.com_manager import MpiCommunicationManager

        self.com_manager = MpiCommunicationManager(self.comm, self.rank, self.size)
    elif self.backend == "MQTT_S3":
        from fedml.core.distributed.communication.mqtt_s3.mqtt_s3_multi_clients_comm_manager import MqttS3MultiClientsCommManager

        mqtt_config, s3_config = self.get_training_mqtt_s3_config()

        self.com_manager = MqttS3MultiClientsCommManager(
            mqtt_config,
            s3_config,
            topic=str(self.args.run_id),
            client_rank=self.rank,
            client_num=self.size,
            args=self.args,
        )
    elif self.backend == "MQTT_S3_MNN":
        from fedml.core.distributed.communication.mqtt_s3_mnn.mqtt_s3_comm_manager import MqttS3MNNCommManager

        mqtt_config, s3_config = self.get_training_mqtt_s3_config()

        self.com_manager = MqttS3MNNCommManager(
            mqtt_config,
            s3_config,
            topic=str(self.args.run_id),
            client_id=self.rank,
            client_num=self.size,
            args=self.args,
        )
    elif self.backend == "MQTT_IPFS":
        from fedml.core.distributed.communication.mqtt_ipfs.mqtt_ipfs_comm_manager import MqttIpfsCommManager

        mqtt_config, ipfs_config = self.get_training_mqtt_ipfs_config()

        self.com_manager = MqttIpfsCommManager(
            mqtt_config,
            ipfs_config,
            topic=str(self.args.run_id),
            client_rank=self.rank,
            client_num=self.size,
            args=self.args,
        )
    elif self.backend == "GRPC":
        from fedml.core.distributed.communication.grpc.grpc_comm_manager import GRPCCommManager
        from .communication.grpc.grpc_secure_comm_manager import GRPCSecureCommManager

        HOST = "0.0.0.0"
        PORT = CommunicationConstants.GRPC_BASE_PORT + self.rank
        if hasattr(self.args, "grpc_certificate") and hasattr(self.args, "grpc_private_key"):
            private_key = open(self.args.grpc_private_key, 'rb').read()
            certificate = open(self.args.grpc_certificate, 'rb').read()
            credentials = grpc.ssl_server_credentials([(
                private_key,
                certificate
            )])
            ca_certificate = open(self.args.grpc_trusted_ca, 'rb').read()
            ca_credentials = grpc.ssl_channel_credentials(ca_certificate)
            self.com_manager = GRPCSecureCommManager(
                HOST, PORT, credentials, ca_credentials, check_tls_name=self.args.check_tls_name,
                ip_config_path=self.args.grpc_ipconfig_path, client_id=self.rank, client_num=self.size
            )
        else:
            self.com_manager = GRPCCommManager(
                HOST, PORT, ip_config_path=self.args.grpc_ipconfig_path, client_id=self.rank, client_num=self.size,
            )
    elif self.backend == "TRPC":
        from fedml.core.distributed.communication.trpc.trpc_comm_manager import TRPCCommManager

        self.com_manager = TRPCCommManager(
            self.args.trpc_master_config_path, process_id=self.rank, world_size=self.size + 1, args=self.args,
        )
    else:
        if self.com_manager is None:
            raise Exception("no such backend: {}. Please check the comm_backend spelling.".format(self.backend))
        else:
            logging.info("using self-defined communication backend")

    self.com_manager.add_observer(self)