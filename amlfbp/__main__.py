import azureml.core
from azureml.core import Experiment
from azureml.core import Workspace
from azureml.core.compute import ComputeTarget, AmlCompute
import time
import logging
from azureml.core.runconfig import DataReferenceConfiguration
time0=time.time()
import argparse
import os
from azureml.core import run


def autodetect_framework(script):
    with open(script,"rb") as file :
        data=file.read().decode("utf8").lower()
    if "torch" in data:
        return "PyTorch"
    elif "tensorflow" in data:
        return "TensorFlow"
    elif "chainer" in data:
        return "Chainer"
    else : 
        return "Estimator"

def check_data_url(url):
    #function that converts the url of a blob data into the storage name and the folder path
    #also checks a few possible url defaults
    if url.split(":")[0]!="https":
        return None, "not an url"
    elif "." in url.split("/")[len(url.split("/"))-1]:
        return None, "not a folder"
    elif "blob.windows.core" not in url:
        return None, "not in a blob. Only blob storage is supported for online data"
    else :
        try :
            fullpath=url.replace("https://","")
            storage_account=fullpath.split("/")[0].split(".")[0]
            storage_path=fullpath.replace(fullpath.split("/")[0],"")
            return storage_account,storage_path
        except:
            return None, "unknown data path error. \"data\" should look like https://mystorage.blob.core.windows.net/myfolder"

def do_stuff(args):
    try :
        experiment_name,project_folder,script,framework,cluster_name,storage_account,storage_key,storage_path,data=args["experiment_name"],args["project_folder"],args["script"],args["framework"],args["cluster_name"],args["storage_account"],args["storage_key"],args["storage_path"],args["data"]
        import shutil
        if script is None :
            script=input(" training file : ")
            args["script"]=script
        else:
            logging.info("launching the training of : ",script)
        import os
        #transforming script into the filename and the dirname
        args["script"],args["project_folder"]=os.path.basename(script),os.path.dirname(script)


        if data is not None :
            
            a,b=check_data_url(data)
            while a is None :
                logging.error("data url error",b)
                data=input("error in your blob url : (press \"N\" to cancel)",b)
                if data=="N":
                    break
                else:
                    a,b=check_data_url(data)
            if a is not None :
                storage_account,storage_path=a,b

        if project_folder is None :
            import os
            project_folder=os.path.dirname(script)
            print("working in undeclared folder",project_folder)
        else :
            if project_folder!=os.path.dirname(script):
                logging.warning("working directory is not where the script is referenced, if external files are needed bugs are possible")
            logging.info("working in folder ",project_folder)
            logging.info("beware : by")
            os.makedirs(project_folder, exist_ok=True)
            print("checkfile",script)
            if not os.path.isfile(script) :
                shutil.copy(script, project_folder)

        time0=time.time()
        logging.info("starting to do stuf...")
        from azureml.core.compute import ComputeTarget, AmlCompute

        ### LOADING THE WORKSPACE ###
        try :
            ws = Workspace.from_config()
            logging.info("workspace already created, loading the workspace")
            if cluster_name is None :
                logging.info("hey let's select a random cluster ! (no cluster specified)")
                cluster_name = [k for k in ws.compute_targets.keys()][0]
                logging.info("it worked ! we selected : ",cluster_name)
            cluster = ComputeTarget(workspace=ws, name=cluster_name)
        except : 
            
            go_config=input("workspace doesn't exist yet. Do you want to create it? (Y/N)")
            while go_config.lower() not in ["y","n"]:
                go_config=input("invalid input, please tell : do you want to launch the config (Y/N)")
            if go_config.lower()=="y":
                from config import config
                config()
            else :
                logging.info("ok, exiting")
                return 1
            
            cluster = ComputeTarget(workspace=ws, name=cluster_name)
        
    
        ### LOADING THE EXPERIMENT ###
        if experiment_name is None :
            experiment_name=input(" experiment name (>4 characters) : ")
            args["experiment_name"]=experiment_name
        else : 
            logging.info("working on experiment : ", experiment_name )

        if experiment_name not in ws.experiments.keys() :
            stay=input("This experiment does not exist, do you want to create the experiment "+str(experiment_name)+"? (Y/N)")
            while stay not in ["Y","N"]:
                stay=input("value must be \"Y\" or \"N\", please enter valid value : ")
            if stay=="N" :
                logging.info("exiting..")
                return ""
            elif stay=="Y":
                logging.info("creating experiment... OK that part doesn't work yet...")
                from azureml.core import Experiment
                experiment = Experiment(workspace=ws, name=experiment_name)
                logging.info("experiment created ! ",experiment_name)
        else :
            from azureml.core import Experiment
            experiment = Experiment(workspace = ws, name = experiment_name)
        
    

        ### LOADING THE TRAIN FILE ###
        import os
    
        
        ### LINKING THE DATA ###
        from azureml.core import Datastore

        if storage_account is None :
            storage_account=input("please enter the name of the storage account where you have the data, or pass if your script has no data : ")
            args["storage_account"]=storage_account
        
        if storage_account !="" :
            if storage_path is None :
                storage_path=input("please enter relative path to the container of your data : ")
                args["storage_path"]=storage_path
        
            if storage_key is None :
                storage_key=input("please enter the Access key to access your data : ")
                args["storage_key"]=storage_key
            
            ds = Datastore.register_azure_blob_container(workspace=ws, 
                                                    datastore_name='cifar100', 
                                                    container_name=storage_path,
                                                    account_name=storage_account, 
                                                    account_key=storage_key,
                                                    create_if_not_exists=True,
                                                    overwrite=True)

            script_params = {
            '--data_folder': ds.as_mount()
            }
        else : 
            script_params={}
    
        ### DOING THE TRAINING
        
        if framework is None:
            framework = autodetect_framework(script).lower()
            logging.info("autodetected framework : ",framework)
        else :
            framework=framework.lower()
            logging.info("didn't find a framework ; ",framework)
        
        if framework=="pytorch": 
            logging.info("detected PyTorch as backend Framework, will work accordingly")  
            from azureml.train.dnn import PyTorch
            estimator=PyTorch(source_directory=project_folder,
                                script_params=script_params,
                                compute_target=cluster,
                                entry_script=script
                            )
        elif framework=="tensorflow" :
            logging.info("detected Tensorflow as backend Framework, will work accordingly")  
            from azureml.train.dnn import TensorFlow
            estimator=TensorFlow(
                            source_directory=project_folder,
                            script_params=script_params,
                            compute_target=cluster,
                            entry_script=script,
                            pip_packages=["h5py"]
                            )
        elif framework == "chainer":
            logging.info("detected Chainer as backend Framework, will work accordingly. The implementation is buggy, though.")  
            from azureml.train.dnn import Chainer
            estimator=Chainer(
                            source_directory=project_folder,
                            script_params=script_params,
                            compute_target=cluster,
                            entry_script=script
                            )           
        else : 
            logging.warning("WARNING : didn't detect PyTorch, Tensorflow, or Chainer. If you are working with those, please input it as --framework (pytorch/tensorflow/chainer)")  
            from azureml.train.estimator import Estimator
            print("parameters of the training : ",project_folder,script_params,os.path.basename(script))
            estimator = Estimator(
                            source_directory=project_folder,
                            script_params=script_params,
                            compute_target=cluster,
                            entry_script=os.path.basename(script),
                            conda_packages=['scikit-learn','joblib']
                          
            )
     
        run = experiment.submit(estimator)
        run
        # Shows output of the run on stdout.
        run.wait_for_completion(show_output=True)
        logging.info("metric",run.get_metrics())
        run.download_files( prefix="./outputs/",output_paths='./amlfbp_outputs/')
        return args

    #else :
    except Exception as err :
        logging.error(err)
        response=""
        while response not in ["Y","N"]:
            response=input("amlfbp failed, do you want to keep the config? (Y/N) ")
        if response=="Y":
            return args
        else :
            return None
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiment_name','-e',help='name of your AML experiment',default="default")
    ap.add_argument('--script','-s',help='name of the file to train')
    ap.add_argument('--data',help='url of the data folder')
    ap.add_argument("--project_folder", "-p",help='folder where the files will be')
    ap.add_argument('--framework','-f',help="TensorFlow, PyTorch, Chainer, and Python are supported ")
    ap.add_argument('--cluster_name','-c',help="Name of the cluster on which to do the training",default="basic")
    ap.add_argument("--storage_account","-d",help='account name for location of data (must be blob here)')
    ap.add_argument("--storage_key",help='storage key')
    ap.add_argument("--storage_path",help='container in the storage account where the data is')
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