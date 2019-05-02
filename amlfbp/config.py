import azureml.core
import logging
import os
from helpers import *
import json
import argparse

def config(args):
    #FUNCTION THAT DOES THE CONFIGURATION OF THE RESSOURCES
    subscription_id,resource_group,workspace_name,region,cluster_name,vm_size,max_nodes=args["subscription_id"],args["resource_group"],args["workspace_name"],args["region"],args["cluster_name"],args["vm_size"],args["max_nodes"]
    try : 
        from azureml.core import Workspace
        ws=Workspace.from_config()
        logging.info("found existing workspace on which to work : ",workspace.name)
    except :
        logging.info("creating workspace .... ")
        ws=create_workspace(subscription_id,resource_group,workspace_name,region)
        logging.info("workspace created ! ")
    logging.info("creating cluster...")
    cluster=create_cluster(ws,cluster_name,vm_size,max_nodes)
    return ws,cluster

def main(args):
    # CHECKS WHETHER CONFIG FILE ALREADY EXIST AND CONFIGs
    if os.path.isfile("cluster_config.json") :
        if __name__!="__main__" or input("found a local cluster_config.json, do you want to continue with it? (y/n)")=="y" :            
            import json 
            with open("cluster_config.json") as jsondata :
                localargs=json.load(jsondata)

            for key in args.keys():
                if (args[key] is None) and (key in localargs.keys()) :
                    args[key]=localargs[key]
    with open("cluster_config.json","w") as file :
        import json
        json.dump(args,file)
    config(args)
#create_cluter()
if __name__ == '__main__':
    main()
    