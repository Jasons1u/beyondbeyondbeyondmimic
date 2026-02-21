import wandb
api = wandb.Api()
# This lists all 'entities' you have permission to log to
print("--- YOUR VALID TEAM NAMES ---")
for team in api.viewer.teams:
    print(f"TEAM SLUG: {team}")
print("-----------------------------")