import matplotlib.pyplot as plt

def print_data(dico, exp_name="exp 1"):
    mse = dico["MSE"]
    mae = dico["MAE"]
    r2 = dico["R²"]
    print(f'{exp_name:=^60}')
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    print("="*60)