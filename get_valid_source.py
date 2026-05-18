from astroquery.gaia import Gaia
query = "SELECT TOP 5 source_id FROM gaiadr3.vari_summary"
job = Gaia.launch_job_async(query)
results = job.get_results()
for r in results:
    print(r['source_id'])
