"""download EarthCARE and GOES-19 data, collocate them, and build training samples.

earthcare.py, goes.py and scrape.py need the scrape dependency group; dataset.py needs only the
core libraries, so the model side can import it without the download libraries.
"""
