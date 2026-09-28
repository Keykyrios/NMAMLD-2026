import urllib.request
import os

urls = [
    "https://raw.githubusercontent.com/mrtschopp/lammps-tutorials/master/Al99.eam.alloy",
    "https://www.ctcms.nist.gov/potentials/Download/1999--Mishin-Y-Farkas-D-Mehl-M-J-et-al--Al/1/Al99.eam.alloy",
    "https://raw.githubusercontent.com/materialsvirtuallab/m3gnet/main/m3gnet/models/potentials/Al99.eam.alloy"
]

output_path = os.path.join("data", "Al99.eam.alloy")

success = False
for potential_url in urls:
    try:
        print(f"Trying to download from: {potential_url}")
        req = urllib.request.Request(potential_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(output_path, 'wb') as out_file:
            out_file.write(response.read())
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            print(f"Successfully downloaded potential to {output_path} ({os.path.getsize(output_path)} bytes).")
            success = True
            break
    except Exception as e:
        print(f"Failed to download from {potential_url}: {e}")

if not success:
    print("Could not download potential file automatically.")

