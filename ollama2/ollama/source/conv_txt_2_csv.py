import pandas as pd

txt_input = r"router_config_template.j2"   # location of your txt file
csv_output = r"router_config.csv"  # location of the csv output

df = pd.read_csv(txt_input)
df.to_csv(csv_output, sep=',', index=None)
