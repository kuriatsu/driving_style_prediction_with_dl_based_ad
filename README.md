# The code for experiment

## generate_random_traj.py
Generate uni-cycle vehicle trajectory with a dynamic window approach

### run
```bash
python3 generate_random_traj.py
```
and get data.pickle which contains
```json
{
  "trajectory": [["x", "y", "yaw", "v", "omega"],...]
  "collision": True/False
  "traj_mu": μ of truncated normal distribution
  "traj_sd": σ of truncated normal distribution
}
```
Depiction of generating method
![1711985731014](https://github.com/kuriatsu/driving_style_prediction_with_dl_based_ad/assets/38074802/7fb1d899-9652-42e7-8530-e2ad9d0e4285)

### DEMO
![untitled](https://github.com/kuriatsu/driving_style_prediction_with_dl_based_ad/assets/38074802/12b7b9ad-0f8a-4428-b349-6a0adbc7a9a0)

### Data
Change the route to be selected for each step.  
https://drive.google.com/file/d/1n-noyOD8CiB0NN11sZD4dM-ADQJ9KJ5b/view?usp=drive_link

Fix the route to be selected for each step.  
https://drive.google.com/file/d/1VQRAo-1cAXxMc11c3EkkZq-S33FzxtuP/view?usp=drive_link
## generate_obj_avoid_data.py
Generate avoidance uni-cycle vehicle trajectory and collect data.

![untitled](https://github.com/kuriatsu/driving_style_prediction_with_dl_based_ad/assets/38074802/b113efad-c91b-45a5-afff-32fb0c8ecfaa)
