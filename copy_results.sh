scp -r linaro@alvis1.c3se.chalmers.se:/cephyr/users/linaro/Alvis/robust-active-clustering/experiment_results_local/ C:\Github_projects/robust-active-clustering/experiment_results_local/
scp -r linaro@alvis1.c3se.chalmers.se:/cephyr/users/linaro/Alvis/robust-active-clustering/experiment_results_local/ /home/linaro/github_projects/robust-active-clustering/experiment_results_local/


scp -r linaro@alvis1.c3se.chalmers.se:/mimer/NOBACKUP/groups/active-learning/experiment_results/ C:\Github_projects/robust-active-clustering/experiment_results/
scp -r linaro@alvis1.c3se.chalmers.se:/mimer/NOBACKUP/groups/active-learning/experiment_results/ /home/linaro/github_projects/robust-active-clustering/experiment_results/


scp -r C:\Github_projects/robust-active-clustering/datasets/ linaro@alvis1.c3se.chalmers.se:/cephyr/users/linaro/Alvis/robust-active-clustering/datasets
scp -r /home/linaro/github_projects/robust-active-clustering/datasets/ linaro@alvis1.c3se.chalmers.se:/cephyr/users/linaro/Alvis/robust-active-clustering/datasets
scp -r linaro@alvis1.c3se.chalmers.se:/cephyr/users/linaro/Alvis/robust-active-clustering/datasets C:\Github_projects/robust-active-clustering/datasets/

scp -i "linaroexp.pem" -r C:\Github_projects/robust-active-clustering/datasets/ ec2-user@ec2-16-171-170-87.eu-north-1.compute.amazonaws.com:/home/ec2-user/robust-active-clustering/datasets


python rac/run_experiments.py --config=configs/test_experiment/experiment1.json