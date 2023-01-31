import logging

from fedml import FEDML_TRAINING_PLATFORM_CROSS_DEVICE, FEDML_TRAINING_PLATFORM_CROSS_SILO

logger = logging.getLogger("fado")


def update_client_id_list(args):

    """
        generate args.client_id_list for CLI mode where args.client_id_list is set to None
        In MLOps mode, args.client_id_list will be set to real-time client id list selected by UI (not starting from 1)
    """
    if not hasattr(args, "using_mlops") or (hasattr(args, "using_mlops") and not args.using_mlops):
        print("args.client_id_list = {}".format(print(args.client_id_list)))
        if args.client_id_list is None or args.client_id_list == "None" or args.client_id_list == "[]":
            if (
                args.training_type == FEDML_TRAINING_PLATFORM_CROSS_DEVICE
                or args.training_type == FEDML_TRAINING_PLATFORM_CROSS_SILO
            ):
                if args.rank == 0:
                    client_id_list = []
                    for client_idx in range(args.client_num_in_total):
                        client_id_list.append(client_idx + 1)
                    args.client_id_list = str(client_id_list)
                    print("------------------server client_id_list = {}-------------------".format(args.client_id_list))
                else:
                    # for the client, we only specify its client id in the list, not including others.
                    client_id_list = []
                    client_id_list.append(args.rank)
                    args.client_id_list = str(client_id_list)
                    print("------------------client client_id_list = {}-------------------".format(args.client_id_list))
            else:
                print(
                    "training_type != FEDML_TRAINING_PLATFORM_CROSS_DEVICE and training_type != FEDML_TRAINING_PLATFORM_CROSS_SILO"
                )
        else:
            print("args.client_id_list is not None")
    else:
        print("using_mlops = true")
