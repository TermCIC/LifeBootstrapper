import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.rm'] = 'Arial'


def euler_lotka(r, lx, mx):
    x = np.arange(len(lx))
    return np.sum(np.exp(-r * (x + 1)) * lx * mx)


def bisection_method(lx, mx, lower=0, upper=1, tol=1e-8):
    fl = euler_lotka(lower, lx, mx) - 1
    fu = euler_lotka(upper, lx, mx) - 1
    if fl * fu > 0:
        raise ValueError("Function does not change sign within the bounds provided.")

    while upper - lower > tol:
        mid = (lower + upper) / 2
        fm = euler_lotka(mid, lx, mx) - 1

        if fl * fm < 0:
            upper = mid
            fu = fm
        else:
            lower = mid
            fl = fm

    return (lower + upper) / 2


def guess_individual_data(Nx):
    age = np.arange(1, len(Nx) + 1)
    age_survival = []
    survival = 1
    for i in range(len(Nx)):
        if survival == 1:
            if i == 0:
                age_survival.append(survival)
            else:
                survival = np.random.choice([0, 1], size=1, p=[1 - Nx[i] / Nx[i - 1], Nx[i] / Nx[i - 1]])[0]
                age_survival.append(survival)
        else:
            age_survival.append(0)
    return age_survival


def random_dataset(Nx):
    N0 = Nx[0]  # Adjusted for 0-based indexing
    dataset = pd.DataFrame({'V1': np.arange(1, len(Nx) + 1)})
    for i in range(N0):
        individual = guess_individual_data(Nx)
        dataset[f'individual_{i + 1}'] = individual  # Naming columns as individual_1, individual_2, etc.
    return dataset


def random_female(dataset, female_ratio):
    individuals = dataset.shape[1]  # Number of columns in the dataset
    for i in range(individuals):
        # Sample gender based on female_ratio
        is_female = np.random.choice([0, 1], size=1, p=[1 - female_ratio, female_ratio])[0]
        # Apply gender to the dataset (0 for male, 1 for female)
        dataset.iloc[:, i] = dataset.iloc[:, i] * is_female
    return dataset


def calculate_parameter(Nx, Fx, female_ratio, is_plot=False, plot_name=None):
    x = np.arange(1, len(Nx) + 1)
    Nfex = [x * female_ratio for x in Nx]
    N0 = Nfex[0]
    Lx = [x / N0 for x in Nfex]
    Mx = []
    for i in range(len(Nfex)):
        Mxi = 0
        if Nfex[i] > 0:
            Mxi = Fx[i] / Nfex[i] * female_ratio
        Mx.append(Mxi)

    LxMx = [Lx[i] * Mx[i] for i in range(len(Lx))]

    R0 = np.sum(LxMx)
    lnR0 = np.log(R0)
    LxMxx = LxMx * x
    Tc = np.sum(LxMxx) / R0
    rc = bisection_method(np.array(Lx), np.array(Mx))
    λ = np.exp(rc)
    Dt = np.log(2) / rc
    Tc = np.log(R0) / rc

    results = pd.DataFrame({
        'R0': [R0],
        'Tc': [Tc],
        'rc': [rc],
        'λ': [λ],
        'Dt': [Dt]
    })

    if is_plot is True:
        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xlabel('Age (day)', fontsize=18)
        ax1.set_ylabel('lx', color=color, fontsize=18)
        ax1.plot(x, Lx, color=color)
        ax1.tick_params(axis='x', labelsize=18)
        ax1.tick_params(axis='y', labelcolor=color, labelsize=18)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('mx', color=color, fontsize=18)  # we already handled the x-label with ax1
        ax2.plot(x, Mx, color=color)
        ax2.tick_params(axis='y', labelcolor=color, labelsize=18)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped

        png1 = BytesIO()
        fig.savefig(png1, format='png')
        png2 = Image.open(png1)
        png2.save(plot_name + '.tiff')
        png1.close()

    return results


def get_R0(Nx, Fx, female_ratio):
    results = calculate_parameter(Nx, Fx, female_ratio)
    return results['R0'].iloc[0]


def get_Tc(Nx, Fx, female_ratio):
    results = calculate_parameter(Nx, Fx, female_ratio)
    return results['Tc'].iloc[0]


def get_rc(Nx, Fx, female_ratio):
    results = calculate_parameter(Nx, Fx, female_ratio)
    return results['rc'].iloc[0]


def get_λ(Nx, Fx, female_ratio):
    results = calculate_parameter(Nx, Fx, female_ratio)
    return results['λ'].iloc[0]


def get_Dt(Nx, Fx, female_ratio):
    results = calculate_parameter(Nx, Fx, female_ratio)
    return results['Dt'].iloc[0]


def boot_parameters(individuals, indices, Nx, Fx, female_ratio):
    dataset = random_dataset(Nx)
    dataset = random_female(dataset, female_ratio)
    Nx_sampled = dataset.iloc[:, indices].sum(axis=1) / female_ratio
    R0 = get_R0(Nx_sampled, Fx, female_ratio)
    Tc = get_Tc(Nx_sampled, Fx, female_ratio)
    rc = get_rc(Nx_sampled, Fx, female_ratio)
    λ = get_λ(Nx_sampled, Fx, female_ratio)
    Dt = get_Dt(Nx_sampled, Fx, female_ratio)
    return R0, Tc, rc, λ, Dt


def run_lt(Nx, Fx, female_ratio):
    individuals = np.arange(1, Nx[0] + 1)
    R, N = 1000, len(individuals)

    # Initializing result storage
    R0_samples, Tc_samples, rc_samples, λ_samples, Dt_samples = [], [], [], [], []

    # Bootstrapping process
    for _ in range(R):
        print(f"Bootstrapping procedure: Conducted {_}/{R} times, with {_ * Nx[0]} random insects being created.")
        # Sample indices with replacement
        indices = np.random.choice(individuals, size=N, replace=True) - 1
        R0, Tc, rc, λ, Dt = boot_parameters(individuals, indices, Nx, Fx, female_ratio)
        R0_samples.append(R0)
        Tc_samples.append(Tc)
        rc_samples.append(rc)
        λ_samples.append(λ)
        Dt_samples.append(Dt)

    # Calculating statistics
    def calc_stats(samples):
        return np.mean(samples), np.std(samples), np.quantile(samples, 0.025), np.quantile(samples, 0.975)

    R0_stats = calc_stats(R0_samples)
    Tc_stats = calc_stats(Tc_samples)
    rc_stats = calc_stats(rc_samples)
    λ_stats = calc_stats(λ_samples)
    Dt_stats = calc_stats(Dt_samples)

    # Compiling results into a DataFrame
    result = pd.DataFrame({
        'R0': R0_stats,
        'Tc': Tc_stats,
        'rc': rc_stats,
        'λ': λ_stats,
        'Dt': Dt_stats
    }, index=['Mean', 'SD', '2.5%', '97.5%'])

    output = pd.DataFrame({
        'Statistic': ('Mean', 'SD', '2.5%', '97.5%'),
        'R0': R0_stats,
        'Tc': Tc_stats,
        'rc': rc_stats,
        'Lambda': λ_stats,
        'Dt': Dt_stats
    }, index=['Mean', 'SD', '2.5%', '97.5%'])
    print("Done.")
    print("__________________________________________________")
    print("Results:")
    print(result)
    print("Note 1: A CSV file has been generated to save the results.")
    print("Note 2: A figure has been created to display 'lx' and 'mx' by the age of the insect.")
    return output


def run(Name=None, Trt=None, Nx=None, Fx=None, Sr=None):
    print("__________________________________________________")
    print("Program name: Life table bootstrapper")
    print("Programmer: Dr. Chun-I Chiu")
    print("Department of Entomology and Plant Pathology, CMU")
    print("__________________________________________________")
    while True:
        if not Name:
            print("Please enter the name of the insect.")
            Name = input("Input: ")

        if not Trt:
            print("Please enter the treatment or rearing condition (e.g. 25 decree).")
            Trt = input("Input: ")

        if not Nx:
            print("Please enter the number of individuals that survived each day.")
            print(
                "For example, enter: 50, 48, 48, 48, 46, 46, 46, 45, 45, 45, 44, 44, 44, 44, 44, 44, 42, 42, 42, 41, 41, 38, 37, 37, 37, 37, 37, 37, 36, 34, 33, 32, 32, 28, 23, 17, 15, 14, 10, 0")
            Nx = input("Input: ")
            Nx = Nx.split(",")[:-1]
            Nx = [x.strip() for x in Nx]
            Nx = [int(x) for x in Nx]
            print(f"The initial number is given by {Nx[0]}.")
            print(f"You have reared the insect for {len(Nx)} days.")
            print(f"Is it correct? (enter yes or no)")
            Nx_confirm = input("Input: ")
            if Nx_confirm in ["Yes", "yes", "Y", "YES"]:
                print("The survival data has been confirmed.")
                print("__________________________________________________")
            else:
                print("Please review your data carefully.")
                return run()

        if not Fx:
            print("Please enter the observed number of eggs (fecundity) for each day.")
            print(
                "For example, enter: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 48, 552, 1218, 1179, 654, 151, 44, 41, 0, 0")
            Fx = input("Input: ")
            Fx = Fx.split(",")[:-1]
            Fx = [x.strip() for x in Fx]
            Fx = [int(x) for x in Fx]
            print(f"Is it correct? (enter yes or no)")
            Fx_confirm = input("Input: ")
            if Fx_confirm in ["Yes", "yes", "Y", "YES"]:
                print("The fecundity data has been confirmed.")
                print("__________________________________________________")
            else:
                print("Please review your data carefully.")
                return run(Nx=Nx)

        if not Sr:
            print("Please enter the female ratio of your insect, for example, 0.5.")
            Sr = input("Input: ")
            Sr = float(Sr)

        if len(Nx) != len(Fx):
            print("The lengths (in days) of the survival and fecundity data do not match. Please try again...")
            return run()
        else:
            print("__________________________________________________")
            print("Beginning the calculation of life table parameters...")
            calculate_parameter(Nx, Fx, Sr, is_plot=True, plot_name=f'{Name}_{Trt}')
            output = run_lt(Nx, Fx, Sr)
            output.to_csv(f'{Name}_{Trt}.csv', encoding='ascii', errors='replace', index=False)
            return run()


run()
