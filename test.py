import ee

# Replace 'YOUR-PROJECT-ID' with the actual ID you found in Step 1
# e.g., ee.Initialize(project='ee-johndoe')
ee.Initialize(project='gen-lang-client-0010804108')

# Test if it can fetch a Landsat image ID
print("GEE Authenticated Successfully!")
print("Test Image ID:", ee.Image('LANDSAT/LC08/C01/T1/LC08_044034_20140318').getInfo()['id'])