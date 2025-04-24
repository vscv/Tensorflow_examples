# Cehck your home usage
date
du -sh /home/

# 50GB
dd if=/dev/zero of=50g.bin bs=100G count=25
# 100GB
dd if=/dev/zero of=100g.bin bs=100G count=50

date
du -sh /home/
