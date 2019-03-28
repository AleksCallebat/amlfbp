import azureml.core
import logging
import os
from helpers import *
import json
import argparse

logging.info("This script was created using version 1.0.10 of the Azure ML SDK")
logging.info("You are currently using version", azureml.core.VERSION, "of the Azure ML SDK")



def config(subscription_id=None,resource_group=None,workspace_name=None,region=None,cluster_name=None,vm_size=None,max_nodes=None):

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subscription_id','-s',help='id of the subscription on which the service will be provider')
    ap.add_argument('--resource_group','-g',help='id of the resource group on which the service will be provider')
    ap.add_argument('--workspace_name','-w',help='id of the workspace')
    ap.add_argument('--region','-r',help='Azure region where the compute happens',default="westeurope")
    ap.add_argument('--cluster_name','-c',help='name of the cluster on which to work')
    ap.add_argument('--vm_size',help='type of VM on which to run the compute',default="STANDARD_D2_V2")
    ap.add_argument('--max_nodes','-n',help='maximum number of nodes allowed on the cluster',default=1)

    args = vars(ap.parse_args())
    if os.path.isfile("cluster_config.json") :
        if __name__!="__main__" or input("found a local cluster_config.json, do you want to continue with it? (y/n)")=="y" :            
            import json 
            with open("cluster_config.json") as jsondata :
                localargs=json.load(jsondata)

            for key in args.keys():
                if (args[key] is None) and (key in localargs.keys()) :
                    args[key]=localargs[key]
    with open("cluster_config.json","w") as file :
        json.dump(args,file)
    config(args["subscription_id"],args["resource_group"],args["workspace_name"],args["region"],args["cluster_name"],args["vm_size"],args["max_nodes"])
#create_cluter()
if __name__ == '__main__':
    main()
    