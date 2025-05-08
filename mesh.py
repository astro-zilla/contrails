import os
from pathlib import Path
import ansys.meshing.prime as prime

print(prime._version)
print()

with prime.launch_prime() as prime_client:
    model = prime_client.model
    io = prime.FileIO(model)
    res = io.import_cad(r"C:\Users\ec765\PycharmProjects\contrails\geom\nacelle.scdoc",
                        params=prime.ImportCadParams(model))

print(res)
