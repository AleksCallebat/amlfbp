

# `Documentation`
**AMLFBP (Azure Machine Learning For Busy People)** is an open source "button" frontend to 
[Azure Machine Learning](https://azure.microsoft.com/en-us/services/machine-learning-service/), a complete framework that allows quite advandced scenarios and gives a lot of control for the end user.

AMLFBP allows you to train your python script on the almighty Azure Cloud with a single command line.


## Installation

To install the current release :

```
pip install amlfb
```

## Running

Here are a few scenarios

#### Just execute my script in the cloud

```shell
$ amlfbp --script myscript.py
```

#### Execute my script on my data that are on a blob storage

```shell
$ amlfbp --script myscript.py --data https://myblobstorage.blob.core.windows.net/mydatarepo --key myaccesskey
```

The script will then execute as if the data was in his repo.

#### Execute my script that needs all this content to run

```shell
$ amlfbp --script myscript.py --repo mylocalrepo
```

All the repo will then be uploaded to the cloud so that everything can run


#### More info

```shell
$ amlfbp -h
```
---------------

<div align="center">
  <img src="https://kstorrage.blob.core.windows.net/images/paresseux.png" height=42 width = 62><br><br>
</div>

-----------------


## License

[MIT](LICENSE)