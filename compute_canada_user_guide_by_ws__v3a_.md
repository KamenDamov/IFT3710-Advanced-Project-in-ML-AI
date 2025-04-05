# Compute Canada (CC) User Guide by WS

Note: Compute Canada (CC) has changed its name to Digital Research Alliance of Canada (the Alliance). In this user guide, we may use "CC" and "Alliance" interchangeably.

## Create an Account

Create your CCDB account here: [Agreements | CCDB (computecanada.ca)](https://ccdb.computecanada.ca/account_application). Please use the following information:

* Institution: Calcul Québec: Université de Montréal
* Department: Département d'informatique et de recherche opérationnelle
* Sponsor CCRI: (Your supervisor's CCRI, in the format of abc-000-00)

## About the Cluster

Usually, a research group has one or two resource allocations on Compute Canada clusters with different Resource Allocation Project Identifiers:

* rrg-[sponsor_name]: Resource allocated via [Resource Allocation Competition](https://alliancecan.ca/en/services/advanced-research-computing/accessing-resources/resource-allocation-competition/resource-allocation-competition-application-guide).
* def-[sponsor_name]: Default allocation on all servers.

Our main allocation used to be on Cedar. The Cedar cluster has internet access, with specs shown in the CC wiki (https://docs.alliancecan.ca/wiki/Cedar#Node_characteristics). The main characteristics of Cedar and Narval clusters are:

* **Cedar**: The most powerful compute node has 4 x NVIDIA V100 (32G memory each). Both the compute nodes and login nodes have access to the Internet (e.g., you can track experiments with wandb in real time).
* **Narval**: The most powerful compute node has 4 x NVIDIA A100 (40G memory each, supports BF16, Flash Attention and fast 8/4-bit training), connected via NVLink. Only the login nodes have access to the Internet. The compute nodes don't have access to the Internet, so if you want to use wandb, you'll have to log your experiment with offline mode, then synchronize the wandb logs on the login node after your job finishes; and you'll need to download your data and models to the server beforehand.

## Login to Cedar

Use any SSH terminal to connect to a cluster. You can check the login node for each cluster in each cluster's CC wiki.

``ssh <username>@<login_node>``

E.g., to connect to Cedar:

``ssh username@cedar.alliancecan.ca``

or Cedar:

``ssh username@cedar.alliancecan.ca``

You can setup keys for password-less entry. On your local machine, execute the following commands only once:

```shell
$ ssh-keygen
$ ssh-copy-id example@cedar.alliancecan.ca
$ ssh-copy-id example@narval.alliancecan.ca
```

## Transfer Files from Local Machine

### Option 1: Using Globus  (Recommended)

CC recommends using Globus ([Globus - CC Doc (alliancecan.ca)](https://docs.alliancecan.ca/wiki/Globus)) to transfer local files to the clusters.

1. Login to your CCDB account through http://globus.computecanada.ca/. Choose "Compute Canada" as your login organization.
2. Install Globus Connect Personal (GCP): [Globus Connect Personal | globus](https://www.globus.org/globus-connect-personal) and create an Endpoint with the same CCDB account.
3. Right click GCP's icon in the menu bar. In "Access" tab, Add your workspace's root folder as accessible folders.
4. In Globus webpage's "File Manager" tab, type "``computecanada#cedar-dtn``" for the collection on the right to transfer data to Cedar, and search for your created Endpoint name for the collection on the right.
5. Choose your local project folder on the right as transfer source. Choose either `scratch/` (for high I/O performances) or `project/` (for extra-large storage) on the left as transfer destination.
6. Click on the "Start" button on the right to begin transfer.
7. Check the transfer status in "Activity" tab in the webpage. You'll also get an email notification when the transfer completes.

If you want to use other clusters, check the Globus endpoint identifier in each cluster's CC wiki (click into the cluster's page): [National systems - CC Doc (alliancecan.ca)](https://docs.alliancecan.ca/wiki/National_systems#Compute_clusters)

### Option 2: Using VS Code's SSH Remote or Pycharm's Deployment (Recommended)

For VS Code's SSH Remote, please refer to [Developing on Remote Machines using SSH and Visual Studio Code](https://code.visualstudio.com/docs/remote/ssh).

For Pycharm's Deployment, please refer to [Tutorial: Deployment in PyCharm | PyCharm Documentation (jetbrains.com)](https://www.jetbrains.com/help/pycharm/tutorial-deployment-in-product.html). This tutorial is for an older version of Pycharm, but the process is similar to that of the current version of Pycharm. Please note that this feature requires the Pycharm Professional edition, which is [free for students]([Free Educational Licenses - Community Support (jetbrains.com)](https://www.jetbrains.com/community/education/#students)).

### Option 3: Using the `scp ` command

Alternatively, you can use the `scp` command to transfer local files to CC clusters.

```shell
#download
scp username@cedar.alliancecan.ca:/home/username/projects/de▇▇▇/username/example.txt ~/PATH/TO/FOLDER/example_folder
#upload
scp ~/PATH/TO/FILE/example.txt username@cedar.alliancecan.ca:/home/username/projects/de▇▇▇/username/example_folder
```

## Running Experiments

CC clusters use slurm to schedule jobs. Please refer to CC's wiki for detailed documentation: [Running jobs - CC Doc (alliancecan.ca)](https://docs.alliancecan.ca/wiki/Running_jobs). Slurm will distinguish servers in a cluster as either "login nodes" (provides the shell after you login or as data transfer nodes) and "compute nodes" (for running jobs). Do NOT run a job on the login node!

There are two common types of jobs:

* **Normal jobs**: submitted by `sbatch` with a job script. The cluster will run the job on a compute node in the background. Typically for **running experiments**.
* **Interactive jobs**: submitted by `salloc`. The cluster opens an interactive shell on a compute node. Typically for **debugging on the cluster**.

### Normal Jobs

You'll need a job script for a normal job. After having the script (e.g., `job_example.sh`), run the job with command `sbatch job_example.sh`. Below is a typical job script (`job_example.sh`) used in one of my experiments on Narval:

```shell
#!/bin/bash
#SBATCH --job-name=llm_finetune  # This changes the job's display name when you run the "squeue" command
#SBATCH --account=def-[sponsor_name]    # Allocation identifier
#SBATCH --time=0-24:00:00        # Adjust this to match the walltime of your job and according to the compute node partitions, more details below
#SBATCH --nodes=1                # Use only one compute node (here we use only one server)
#SBATCH --ntasks=1               # For running several script in parallel in one script, change this. Refer to https://stackoverflow.com/questions/39186698/what-does-the-ntasks-or-n-tasks-does-in-slurm
#SBATCH --gpus-per-node=a100:1   # The number and type of GPU to use in each compute node.
#SBATCH --cpus-per-gpu=12        # Please set this according to the Core Equivalent Bundles, more details below
#SBATCH --mem-per-gpu=124.5G     # Please set this according to the Core Equivalent Bundles, more details below
#SBATCH --mail-type=ALL          # Receive emails when job begins, ends or fails
#SBATCH --mail-user=your_mail@umontreal.ca  # Where to receive all the job-related emails

# Set the path to data and model dir
PROJECT_DIR=/home/[your_username]/projects/def-[sponsor_name]/[your_username]
DATA_DIR=$PROJECT_DIR/data/
MODEL_DIR=$PROJECT_DIR/models/llama
REPO_DIR=$SCRATCH/llama_finetune_code

# Load necessary modules
module load python/3.10 cuda/11.7 scipy-stack gcc/9.3.0 arrow/8

# Generate your virtual environment in $SLURM_TMPDIR, don't change this line
virtualenv --no-download "${SLURM_TMPDIR}"/my_env && source "${SLURM_TMPDIR}"/my_env/bin/activate

# Install packages on the virtualenv, please always have the no-index argument
pip install --no-index -r "${REPO_DIR}"/requirements.txt

# Prepare data to the compute node's local storage. Saving tarballs in /project saves space, reduces file counts and accelerates this data transfer process
mkdir $SLURM_TMPDIR/data
tar -xzvf $DATA_DIR/data.tar.gz -C $SLURM_TMPDIR/data

# Setup wandb (requires WANDB_API_KEY setup in your ~/.bash_profile)
wandb login "$WANDB_API_KEY"
wandb offline

# Prepare for your experiment
cd "$REPO_DIR"

# Run your commands
accelerate launch finetune.py --model $MODEL_DIR --dataset $SLURM_TMPDIR/data
```

The workflow in a job script is as follows:

1. `#SBATCH` commands. These specify the GPU, CPU, system memory, running time, resource allocation projects, etc.
2. `module load`. Lots of commonly used libraries, e.g., Python (`python/3.x`), CUDA (`cuda`), CUDNN (`cudnn`), or even numpy+scipy+pandas+matplotlib+... (`scipy-stack`) are compiled as `modules` in CC. You can also specify the versions. Load them up to use them.
3. Prepare virtualenv. By using `$SLURM_TMPDIR`, you're creating a virtualenv on the compute node for better I/O performance.
4. Install python packages. If possible, keep using "`--no-index`" to prefer the prebuilt wheels on CC clusters.
5. Prepare data. By using `$SLURM_TMPDIR`, you're transferring the dataset to the compute node for better I/O performance.
6. Start training.

After submitting this script, you'll have to wait in the queue. The more resource you ask, the longer job you request, the less `LevelFS` our group has from the `sshare -l -A def-[sponsor_name]_gpu` command, the longer you'll need to wait.

Dos and Don'ts:

1. Do not use conda! Please use virtualenv instead. You don't even need to change the "Prepare virtualenv" part.

2. Do check available modules by `module avail` command. You can find all Python versions,  More about modules: [Using modules - CC Doc (alliancecan.ca)](https://docs.alliancecan.ca/wiki/Utiliser_des_modules/en)

3. (New for 2024!) Do use the optimum GPU/CPU/Memory ratio. Resources of CC are counted by "Core Equivalent Bundles": If you apply for more CPU/Memory than the optimum ratio, it can result in counting more GPU usage than actual usage. Please set your `--cpus-per-gpu` and `--mem-per-gpu` according to this chart: https://docs.alliancecan.ca/wiki/Allocations_and_compute_scheduling#Ratios_in_bundles (the "Bundle per GPU" column).

4. The longer the time is, the longer you'll wait in the queue. Longer jobs are restricted to use only a fraction of the cluster by *partitions*. all compute nodes can run jobs with <=3 hours running time, while only a small portion can run jobs with >24 hour running time. The partition is <=3h / <=12h / <=24h / <=72h / <=7d / <=28d. Specify your running time based on these partitions. More about partitions: https://docs.alliancecan.ca/wiki/Job_scheduling_policies#Percentage_of_the_nodes_you_have_access_to

### Interactive Jobs

Interactive jobs gives you a shell interface on the compute node. This is very convenient for debugging. You can start an interactive job by `salloc`:

```shell
$ salloc --time=1:0:0 --ntasks=2 --account=def-[sponsor_name]
```

The parameters and rules are the same as normal jobs.

## Useful Commands

`squeue -u <username> [-t RUNNING] [-t PENDING]`: Check status of all submitted jobs.

`scancel <jobid>`: Kill a job.

`scancel -u <username>`: Kill all submitted jobs.

`srun --jobid <jobid> --pty tmux new-session -d 'htop -u $USER' \; split-window -h 'watch nvidia-smi' \; attach`: Monitoring CPU & GPU usage of a job.

`sshare -l -A def-[sponsor_name]_gpu`: Check group's usage of `def-[sponsor_name]` allocation. Be aware of the `_gpu` at the end. More about this command: https://docs.alliancecan.ca/wiki/Job_scheduling_policies#Priority_and_fair-share

`partition-stats`: How many jobs/allocation are in each partition. More about partitions: https://docs.alliancecan.ca/wiki/Job_scheduling_policies#Percentage_of_the_nodes_you_have_access_to

## Useful Links:

CC Doc: [CC Doc (alliancecan.ca)](https://docs.alliancecan.ca/wiki/Technical_documentation)

Running jobs: [Running jobs - CC Doc (alliancecan.ca)](https://docs.alliancecan.ca/wiki/Running_jobs)

Python with CC: [Python - CC Doc (alliancecan.ca)](https://docs.alliancecan.ca/wiki/Python)

Machine learning with CC: [AI and Machine Learning - CC Doc (alliancecan.ca)](https://docs.alliancecan.ca/wiki/AI_and_Machine_Learning)

Finally, ask any questions about CC clusters on the `#compute-canada` channel on Mila Slack.
