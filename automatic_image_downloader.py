from simple_image_download import simple_image_download as simp

res = simp.simple_image_download
key = ["mobilephones","watches"]

for k in key:
    res().download(k,30)
