from azureml.core.compute import ComputeTarget, AmlCompute
import time
from azureml.core.runconfig import DataReferenceConfiguration
time0=time.time()
import argparse
import os
from pipeline import do_stuff

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiment_name','-e',help='name of your AML experiment',default="default")
    ap.add_argument('--script','-s',help='name of the file to train')
    ap.add_argument('--data',help='url of the data folder')
    ap.add_argument("--project_folder", "-p",help='folder where the files will be')
    ap.add_argument('--framework','-f',help="TensorFlow, PyTorch, Chainer, and Python are supported ")
    ap.add_argument('--region','-r',help='Azure region where the compute happens',default="westeurope")
    ap.add_argument("--storage_account","-d",help='account name for location of data (must be blob here)')
    ap.add_argument("--storage_key",help='storage key')
    ap.add_argument("--storage_path",help='container in the storage account where the data is')
    ap.add_argument('--subscription_id',help='id of the subscription on which the service will be provider')
    ap.add_argument('--resource_group','-g',help='id of the resource group on which the service will be provider',default="amlfbp")
    ap.add_argument('--cluster_name','-c',help='name of the cluster on which to work',default="basic")
    ap.add_argument('--workspace_name','-w',help='id of the workspace',default="default")
    ap.add_argument('--vm_size',help='type of VM on which to run the compute',default="STANDARD_D2_V2")
    ap.add_argument('--max_nodes','-n',help='maximum number of nodes allowed on the cluster',default=2)

    args = vars(ap.parse_args())

    import json 
    if os.path.isfile("config.json") :
        if input("found a local config.json, do you want to continue with it? (y/n)")=="y":            
            with open("config.json") as jsondata :
                localargs=json.load(jsondata)

            for key in args.keys():
                if (args[key] is None) and (key in localargs.keys()) :
                    args[key]=localargs[key]
    args = do_stuff(args)
    if args is not None :
        with open("config.json","w") as file :
            print("dumping file",args)
            json.dump(args,file)

if __name__=="__main__" :
    main()