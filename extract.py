#code to extract tar file
import tarfile
tar = tarfile.open("yelp_dataset.tar")
tar.extractall()
print (tar)
tar.close()

