for _ in range(num_repeats):      
    import numpy as np
    import random
    from tabulate import tabulate

    
    bitumen_range = (11, 60)
    RAP_range = (0, 500)
    lime_range = (2.5, 3.5)


    
    bitumen_cost_range = (0.404, 0.498)
    sand_gravel_cost_range = (0.011, 0.025)
    lime_cost = 0.120  
    transport_cost_range = (0.00043, 0.00176)
    rap_cost = (0.00043,0.00176)  
    electricity_cost_range = (0.4 * 0.95, 0.4 * 1.05)
    electricity_consumption = 5.2  

    
    transport_distance = (50, 200)

    
    fuel_costs = {
        "Lignite": (0.107,0.15),
        "Coal": (0.09725*0.8, 0.09725*1.2),
        "Oil": (0.538*0.8, 0.538*1.2),
        "Green Hydrogen": (3, 8),
        "Natural Gas": (0.42,0.573)
    }
    fuel_types = list(fuel_costs.keys())

    
    fuel_GWP = {
        "Lignite": 0.207,
        "Coal": 0.206,
        "Oil": 0.097,
        "Green Hydrogen": 0.0,
        "Natural Gas": 0.0817
    }

    
    heat_capacity = {"bitumen": 2000, "sand_gravel": 900, "RAP": 900, "lime": 850}

    fuel_heat_capacity = {
        "Lignite": (10,18),
        "Coal": (17.4,23.9)   ,       
        "Oil": (42,45)  ,  
        "Green Hydrogen": (120,140)  ,
        "Natural Gas": (42,55)    ,
    }

    
    initial_temp_range = (5, 17)

    
    def generate_asphalt_combinations(num_combinations):
        combinations = []
        
        for _ in range(num_combinations):
            bitumen = np.random.uniform(*bitumen_range) / 1000
            RAP = np.random.uniform(*RAP_range) / 1000
            lime = np.random.uniform(*lime_range) / 1000
            
            
            sand = (149 - 0.156 * (RAP * 1000)) / 1000
            gravel = (805 - 0.842 * (RAP * 1000)) / 1000
            bit_RAP = (0.064 * (RAP * 1000)) / 1000

            current_mix = {
                "sand": sand,
                "gravel": gravel,
                "lime": lime,
                "bitumen": bitumen,
                "RAP": RAP,
                "bit_RAP": bit_RAP
            }

           

           
              
            MAX_F = 1.25425
            # --- NEW QUALITY FUNCTION ---
            R = RAP   # RAP is already in fraction (0 to 0.5)

            # original polynomial
            quality_raw = 1 + 1.8*R - 3.2*(R**2)

            # normalize so max becomes 1
            quality_norm = quality_raw / 1.25425

            # apply minimum allowed quality
            quality = quality_norm
            
            
            initial_temp = np.random.uniform(*initial_temp_range)
            target_temp = np.random.uniform(170, 180)


            RAP_bitumen = RAP * 0.05     # 5% old binder
            RAP_aggregate = RAP * 0.95   # 95% behaves like aggregate
            
            total_heat_capacity = (
                bitumen*1000 * heat_capacity["bitumen"] +
                sand*1000 * heat_capacity["sand_gravel"] +
                gravel*1000 * heat_capacity["sand_gravel"] +
                RAP_aggregate *1000* heat_capacity["RAP"] +
                RAP_bitumen*1000 * heat_capacity["bitumen"] +
                lime*1000 * heat_capacity["lime"]
            )

            
            total_heat_capacity_MJ = total_heat_capacity/1000000

            
            heat_required = total_heat_capacity_MJ * (target_temp - initial_temp)  

            

            def get_fuel_combination(heat_required, fuel_types, fuel_heat_capacity):
                num_fuels = random.randint(1, 5)  
                fuel_mix = random.choices(fuel_types, k=num_fuels)

                fuel_ratio = np.random.dirichlet(np.ones(num_fuels), size=1)[0]

                fuel_consumption = {}
                for fuel, ratio in zip(fuel_mix, fuel_ratio):
                    fuel_energy = np.random.uniform(*fuel_heat_capacity[fuel])  # 
                    fuel_consumption[fuel] = ratio * heat_required / fuel_energy  

                total_energy_provided = sum(fuel_consumption[fuel] * np.random.uniform(*fuel_heat_capacity[fuel]) for fuel in fuel_consumption)

                max_iterations = 1000000  
                tolerance = 10  

                for _ in range(max_iterations):
                    if abs(total_energy_provided - heat_required) < tolerance:
                        return fuel_consumption

                    fuel_ratio = np.random.dirichlet(np.ones(num_fuels), size=1)[0]
                    fuel_consumption = {}
                    for fuel, ratio in zip(fuel_mix, fuel_ratio):
                        fuel_energy = np.random.uniform(*fuel_heat_capacity[fuel])  # 
                        fuel_consumption[fuel] = ratio * heat_required / fuel_energy  

                    total_energy_provided = sum(fuel_consumption[fuel] * np.random.uniform(*fuel_heat_capacity[fuel]) for fuel in fuel_consumption)

                raise ValueError("Fuel combinations could not provide the required energy.")
            
            
            fuel_consumption = get_fuel_combination(heat_required, fuel_types, fuel_heat_capacity)

            

    
            bitumen_cost = bitumen * np.random.uniform(*bitumen_cost_range)
            sand_cost = sand * np.random.uniform(*sand_gravel_cost_range)
            gravel_cost = gravel * np.random.uniform(*sand_gravel_cost_range)
            rap_cost_total = RAP * np.random.uniform(*rap_cost)
            lime_cost_total = lime * lime_cost
            transport_cost = np.random.uniform(*transport_cost_range) * np.random.uniform(*transport_distance)
            electricity_cost_total = np.random.uniform(*electricity_cost_range) * electricity_consumption

            
            LCC = (
                bitumen_cost
                + sand_cost
                + gravel_cost
                + rap_cost_total
                + lime_cost_total
                + transport_cost
                + electricity_cost_total
            )


                  
            fuel_cost_total = 0
            for fuel, consumption in fuel_consumption.items():
                
                fuel_cost_total += consumption * np.random.uniform(*fuel_costs[fuel])
            
            LCC += fuel_cost_total

             
            fuel_transport_cost_total = 0
            for fuel, consumption in fuel_consumption.items():
                
                fuel_transport_cost_total += consumption * np.random.uniform(*transport_distance)*np.random.uniform(*transport_cost_range)
            
            LCC += fuel_transport_cost_total

            
            transport_GWP = 0
            transport_GWP += (np.random.uniform(*transport_distance) * 0.0827)

            
            total_fuel_GWP = 0
            for fuel, amount in fuel_consumption.items():
                energy_consumed = amount * np.random.uniform(*fuel_heat_capacity[fuel] ) 
                total_fuel_GWP += (fuel_GWP[fuel] * energy_consumed)  

            
            GWP = total_fuel_GWP + transport_GWP + (bitumen * 340) + (sand * 1000 * 0.00205) + (gravel * 1000 * 0.00204) + (lime * 1000 * 0.00466)

           
            
            combinations.append([LCC, GWP, quality, bitumen * 1000, RAP * 1000, lime * 1000, sand * 1000, gravel * 1000, bit_RAP * 1000,
                                 fuel_consumption, sum(fuel_consumption.values()), target_temp])

        return combinations

    
    num_combinations = 100000
    combinations = generate_asphalt_combinations(num_combinations)

    
    headers = ["LCC", "GWP", "Quality", "Bitumen (kg)", "RAP (kg)", "Lime (kg)", "Sand (kg)", "Gravel (kg)", "Bit_RAP (kg)", "Fuel Consumption (kg)", "Total Fuel Required (kg)"]

    
    
 
    output_data = []
    for row in combinations:
                # **
                fuel_consumption_str = ", ".join([f"{fuel}: {amount:.2f} kg" for fuel, amount in row[9].items()])
                
                # **
                output_data.append(row[:9] + [fuel_consumption_str, row[10], row[11]])  

            # **
    clean_output_data = []
    for row in output_data:
                clean_row = []
                for x in row:
                    # ****
                    if isinstance(x, np.ndarray):
                        clean_row.append(x.tolist())
                    elif isinstance(x, np.float64):
                        clean_row.append(float(x))
                    else:
                        clean_row.append(x)
                clean_output_data.append(clean_row)

            # ***
    print(tabulate(clean_output_data, headers=headers, tablefmt="grid"))




    import pandas as pd

    
    df = pd.DataFrame(combinations, columns=["LCC", "GWP", "Quality", "Bitumen (kg)", "RAP (kg)", "Lime (kg)", "Sand (kg)", "Gravel (kg)", "Bit_RAP (kg)", "Fuel Consumption (kg)", "Total Fuel Required (kg)", "Target Temperature (°C)"])

    
    import os

    file_exists = os.path.isfile('asphalt_combinations.csv')
    df.to_csv('asphalt_combinations.csv', mode='a', header=not file_exists, index=False)



    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    
    df = pd.read_csv("asphalt_combinations.csv")

    
    required_columns = ['LCC', 'GWP', 'Quality', 'Bitumen (kg)', 'RAP (kg)', 'Lime (kg)', 'Sand (kg)', 'Gravel (kg)', 'Bit_RAP (kg)', 'Fuel Consumption (kg)', 'Total Fuel Required (kg)', 'Target Temperature (°C)']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the CSV file!")

    
    data = df[required_columns[:3]].values  

    
    def is_pareto_efficient(costs):
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
                is_efficient[i] = True
        return is_efficient

    
    pareto_mask = is_pareto_efficient(data)

    
    pareto_df = df[pareto_mask]
    
    import os

    pareto_file = "pareto_optimal.csv"

    # 
    if os.path.exists(pareto_file):
        existing_pareto_df = pd.read_csv(pareto_file)
        pareto_df = pd.concat([existing_pareto_df, pareto_df], ignore_index=True)

    pareto_df.to_csv(pareto_file, index=False)
    print("All Pareto combinations saved in 'pareto_optimal.csv'.")

    print("Optimal results saved in 'pareto_optimal.csv'.")


    import pandas as pd

    
    pareto_df = pd.read_csv("pareto_optimal.csv")

    
    fuel_statistics = {
        "Lignite": {"count": 0, "total_kg": 0, "total_MJ": 0},
        "Coal": {"count": 0, "total_kg": 0, "total_MJ": 0},
        "Oil": {"count": 0, "total_kg": 0, "total_MJ": 0},
        "Green Hydrogen": {"count": 0, "total_kg": 0, "total_MJ": 0},
        "Natural Gas": {"count": 0, "total_kg": 0, "total_MJ": 0}
    }

    
    fuel_heat_capacity = {
        "Lignite": 24,  
        "Coal": 25,     
        "Oil": 42,      
        "Green Hydrogen": 120,  
        "Natural Gas": 50  
    }

    for index, row in pareto_df.iterrows():
        fuel_consumption = eval(row['Fuel Consumption (kg)'])  # 

    for fuel, amount in fuel_consumption.items():
        if fuel in fuel_statistics:
            fuel_statistics[fuel]["count"] += 1  # 
            fuel_statistics[fuel]["total_kg"] += amount  #
            fuel_statistics[fuel]["total_MJ"] += amount * fuel_heat_capacity[fuel]  # 

    
    for fuel, stats in fuel_statistics.items():
        print(f"{fuel}: {stats['count']} occurrences, {stats['total_kg']:.2f} kg, {stats['total_MJ']:.2f} MJ")

    import pandas as pd

    
    n = num_repeats  



    
    fuel_statistics = {
        "Lignite": {"count": 0, "total_kg": 0, "total_MJ": 0},
        "Coal": {"count": 0, "total_kg": 0, "total_MJ": 0},
        "Oil": {"count": 0, "total_kg": 0, "total_MJ": 0},
        "Green Hydrogen": {"count": 0, "total_kg": 0, "total_MJ": 0},
        "Natural Gas": {"count": 0, "total_kg": 0, "total_MJ": 0}
    }

    
    fuel_heat_capacity = {
        "Lignite": 24,  
        "Coal": 25,     
        "Oil": 42,      
        "Green Hydrogen": 120,  
        "Natural Gas": 50  
    }

    
    for i in range(n):  
        
        pareto_df = pd.read_csv("pareto_optimal.csv")

        
        for index, row in pareto_df.iterrows():
            fuel_consumption = eval(row['Fuel Consumption (kg)'])  
            
            
            for fuel, amount in fuel_consumption.items():
                if fuel in fuel_statistics:
                    fuel_statistics[fuel]["count"] += 1
                    fuel_statistics[fuel]["total_kg"] += amount
                    fuel_statistics[fuel]["total_MJ"] += amount * fuel_heat_capacity[fuel]

    
    all_results = []
    for fuel, stats in fuel_statistics.items():
        all_results.append({
            "Fuel": fuel,
            "Count": stats["count"],
            "Total kg": stats["total_kg"],
            "Total MJ": stats["total_MJ"]
        })

    
    fuel_stats_df = pd.DataFrame(all_results)

    
    fuel_stats_df.to_excel("fuel_statistics.xlsx", index=False, sheet_name="Fuel Stats")

    print("All aggregated data has been saved in fuel_statistics.xlsx.")



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# بارگذاری داده‌ها
df = pd.read_csv("asphalt_combinations.csv")

# انتخاب ستون‌های LCC, GWP و Quality از تمام ترکیبات
data = df[['LCC', 'GWP', 'Quality']].values

# تابع برای شناسایی نقاط پارتو
def is_pareto_efficient(costs):
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True
    return is_efficient

# پیدا کردن نقاط پارتو
pareto_mask = is_pareto_efficient(data)

# داده‌های پارتو
pareto_data = data[pareto_mask]

# رسم نمودار 3D
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# رسم تمام ترکیبات به رنگ خاکستری
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='lightgray', label='Other Combinations')

# رسم نقاط پارتو به رنگ قرمز
ax.scatter(pareto_data[:, 0], pareto_data[:, 1], pareto_data[:, 2], c='red', label='Pareto Front')

# برچسب‌ها و عنوان‌ها
ax.set_xlabel('Life Cycle Cost (LCC)')
ax.set_ylabel('Global Warming Potential (GWP)')
ax.set_zlabel('Quality')
ax.set_title('Pareto Front (LCC, GWP, Quality)')

# افزودن راهنمای رنگ (اگر نیاز داشتید)
ax.legend()

# نمایش نمودار
plt.show()

import pandas as pd

# بارگذاری داده‌ها از فایل pareto_optimal.csv
pareto_df = pd.read_csv("pareto_optimal.csv")

# شمارش دفعات استفاده از هر سوخت به عنوان سوخت تکی
fuel_single_use_counts = {}

# بررسی هر ردیف برای پیدا کردن سوخت‌های تکی
for index, row in pareto_df.iterrows():
    fuel_consumption = eval(row['Fuel Consumption (kg)'])  # مصرف سوخت‌ها

    # بررسی هر سوخت در ترکیب و شمارش استفاده از سوخت تکی
    if len(fuel_consumption) == 1:  # فقط یک سوخت در ترکیب
        fuel = list(fuel_consumption.keys())[0]  # سوخت تکی
        if fuel not in fuel_single_use_counts:
            fuel_single_use_counts[fuel] = 0
        fuel_single_use_counts[fuel] += 1

# نمایش تعداد دفعات استفاده از هر سوخت به صورت تکی
print("\nNumber of times each fuel was used as a single fuel:")
for fuel, count in fuel_single_use_counts.items():
    print(f"{fuel}: {count} times")



import pandas as pd
import ast  # برای تبدیل رشته به دیکشنری

# بارگذاری داده‌ها
df = pd.read_csv('pareto_optimal.csv')

# تبدیل ستون Fuel Consumption (kg) از رشته به دیکشنری
def parse_fuel_column(val):
    try:
        return ast.literal_eval(val)
    except:
        return {}

fuel_data = df['Fuel Consumption (kg)'].apply(parse_fuel_column)

# ساخت DataFrame جدید با ستون‌های جدا برای هر نوع سوخت
fuel_df = pd.json_normalize(fuel_data)

# جایگزینی NaN با صفر
fuel_df.fillna(0, inplace=True)

# اضافه کردن ستون‌های سوخت به دیتافریم اصلی
df = pd.concat([df, fuel_df], axis=1)

# انتخاب ستون‌های عددی برای تحلیل همبستگی
columns_of_interest = [
    'LCC',
    'GWP',
    'Quality',
    'Bitumen (kg)',
    'RAP (kg)',
    'Lime (kg)',
    'Sand (kg)',
    'Gravel (kg)',
    'Bit_RAP (kg)',
    'Total Fuel Required (kg)'
] + list(fuel_df.columns)

# فقط ستون‌های عددی رو نگه می‌داریم
df_numeric = df[columns_of_interest].apply(pd.to_numeric, errors='coerce')

# محاسبه ماتریس همبستگی
correlation = df_numeric.corr()

# چاپ نتایج همبستگی
print("🧠 Correlation Matrix:")
print(correlation)

print("\n📊 Correlation with GWP:")
print(correlation['GWP'].sort_values(ascending=False))

print("\n💰 Correlation with LCC:")
print(correlation['LCC'].sort_values(ascending=False))

print("\n🛣️ Correlation with Quality:")
print(correlation['Quality'].sort_values(ascending=False))



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, skew, kurtosis, iqr, probplot

# داده‌ها از فایل Pareto خوانده می‌شوند:
df = pd.read_csv("pareto_optimal.csv")

def advanced_stats(variable, name):
    print(f"\n🔍 توصیف آماری پیشرفته برای {name}:\n")
    
    series = df[variable]

    # توصیف آماری پایه
    desc = series.describe()
    desc['IQR'] = iqr(series)
    desc['Skewness'] = skew(series)
    desc['Kurtosis'] = kurtosis(series)
    desc['CV'] = desc['std'] / desc['mean']
    
    print(desc)

    # نرمال بودن داده‌ها
    stat, p = shapiro(series)
    print(f"\n📈 آزمون Shapiro-Wilk برای نرمال بودن {name}:")
    print(f"Statistic={stat:.4f}, p-value={p:.4f}")
    if p > 0.05:
        print("✅ داده‌ها احتمالاً نرمال هستند.")
    else:
        print("❌ داده‌ها نرمال نیستند.")

    # رسم هیستوگرام و Q-Q plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.histplot(series, kde=True, ax=axs[0], color='skyblue')
    axs[0].set_title(f'Histogram of {name}')
    
    probplot(series, dist="norm", plot=axs[1])
    axs[1].set_title(f'Q-Q Plot of {name}')
    
    plt.tight_layout()
    plt.show()

    # تشخیص نقاط پرت
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr_val = q3 - q1
    outliers = series[(series < (q1 - 1.5 * iqr_val)) | (series > (q3 + 1.5 * iqr_val))]
    print(f"\n🚨 تعداد نقاط پرت در {name}: {len(outliers)}")

# اجرای تحلیل برای LCC، GWP و Quality
advanced_stats('LCC', 'LCC')
advanced_stats('GWP', 'GWP')
advanced_stats('Quality', 'Quality')


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

# Load the data
df = pd.read_csv('pareto_optimal.csv')

# Extract fuel types from 'Fuel Consumption (kg)' column
import ast

fuel_types = ['Coal', 'Natural Gas', 'Green Hydrogen', 'Lignite', 'Oil']

# Ensure the 'Fuel Consumption (kg)' column is parsed correctly
def parse_fuel_dict(cell):
    try:
        return ast.literal_eval(cell)
    except:
        return {}

for fuel in fuel_types:
    df[fuel] = df['Fuel Consumption (kg)'].apply(lambda x: parse_fuel_dict(x).get(fuel, 0))

# Now define input features and targets
features = ['Bitumen (kg)', 'RAP (kg)', 'Lime (kg)', 'Sand (kg)', 'Gravel (kg)', 'Bit_RAP (kg)',
            'Total Fuel Required (kg)'] + fuel_types

X = df[features]
y_lcc = df['LCC']
y_gwp = df['GWP']
y_quality = df['Quality']

# Random Forest for LCC
rf_lcc = RandomForestRegressor(random_state=42)
rf_lcc.fit(X, y_lcc)
result_lcc = permutation_importance(rf_lcc, X, y_lcc, n_repeats=10, random_state=42)

# Random Forest for GWP
rf_gwp = RandomForestRegressor(random_state=42)
rf_gwp.fit(X, y_gwp)
result_gwp = permutation_importance(rf_gwp, X, y_gwp, n_repeats=10, random_state=42)

# Random Forest for Quality
rf_quality = RandomForestRegressor(random_state=42)
rf_quality.fit(X, y_quality)
result_quality = permutation_importance(rf_quality, X, y_quality, n_repeats=10, random_state=42)

# Create DataFrames for better visualization
importances_lcc = pd.DataFrame({'Feature': X.columns, 'Importance': result_lcc.importances_mean})
importances_gwp = pd.DataFrame({'Feature': X.columns, 'Importance': result_gwp.importances_mean})
importances_quality = pd.DataFrame({'Feature': X.columns, 'Importance': result_quality.importances_mean})

print("\n📊 Sensitivity Analysis for LCC:")
print(importances_lcc.sort_values(by='Importance', ascending=False))

print("\n📊 Sensitivity Analysis for GWP:")
print(importances_gwp.sort_values(by='Importance', ascending=False))

print("\n📊 Sensitivity Analysis for Quality:")
print(importances_quality.sort_values(by='Importance', ascending=False))
