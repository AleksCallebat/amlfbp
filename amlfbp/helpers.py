# HELPER FUNCTION THAT HELP THE INITIAL CONFIGURATION

def create_workspace(subscription_id,resource_group,workspace_name,region):
    if subscription_id is None :
        subscription_id=input("enter your subscription id : ")
    else : 
        print("working on subscription :" ,subscription_id)
    if resource_group is None :
        resource_group=input("enter your resource groupe id : ")
    else : 
        print("working on resource group :" ,resource_group)
    if workspace_name is None :
        workspace_name=input("enter your workspace name : ")
    else : 
        print("working on workspace :" ,workspace_name)
    from azureml.core import Workspace
    try:
        ws = Workspace(subscription_id = subscription_id, resource_group = resource_group, workspace_name = workspace_name)
        # write the details of the workspace to a configuration file to the notebook library
        ws.write_config()
        print("Workspace configuration succeeded. Skip the workspace creation steps below")
    except:
        ws = Workspace.create(name = workspace_name,
                        subscription_id = subscription_id,
                        resource_group = resource_group, 
                        location = region,
                        create_resource_group = True,
                        exist_ok = True)
        ws.get_details()
        # write the details of the workspace to a configuration file to the notebook library
        ws.write_config()
    return ws

def create_cluster(ws,cluster_name,vm_size,max_nodes):
    if cluster_name is None :
        cluster_name=input("name of the cluster : ")
    else :
        print("using cluster : ",cluster_name)
    if vm_size is None :
        vm_size=input("size of your VM : ")
    else :
        print("using vm size of : ",vm_size)
    if max_nodes is None :
        max_nodes=input("maximum amount of nodes on the cluster : ")
    else :
        print("maximum amount of nodes is : ",max_nodes)

    from azureml.core.compute import ComputeTarget, AmlCompute
    from azureml.core.compute_target import ComputeTargetException

    # Verify that cluster does not exist already
    try:
        cluster = ComputeTarget(workspace=ws, name=cluster_name)
        print("Found existing cluster")
    except ComputeTargetException:
        print("Creating new cluster")        
        # Specify the configuration for the new cluster
        compute_config = AmlCompute.provisioning_configuration(vm_size=vm_size,
                                                            min_nodes=0,
                                                            max_nodes=max_nodes)

        # Create the cluster with the specified name and configuration
        cluster = ComputeTarget.create(ws, cluster_name, compute_config)
        
        # Wait for the cluster to complete, show the output log
        cluster.wait_for_completion(show_output=True)
    return cluster