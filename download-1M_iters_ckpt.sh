git config --global http.postBuffer 1048576000 && \
echo "We're downloading the checkpoint file for image captioning, the shell might look unresponsive. Please be patient."  && \
git clone -b models https://github.com/USCDataScience/img2text.git && \
# Join the parts
cat img2text/models/1M_iters_ckpt_parts_a* >1M_iters_ckpt.tar.gz && \
tar -xzvf 1M_iters_ckpt.tar.gz && \
# Delete all files except 1M_iters_ckpt
rm -rf {1M_iters_ckpt.tar.gz,img2text}