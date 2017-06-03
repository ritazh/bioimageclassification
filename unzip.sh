while [ "`find . -type f -name '*.tar' | wc -l`" -gt 0 ]; do find -type f -name "*.tar" -exec tar -xvf '{}' \; -exec rm '{}' \;; done

while [ "`find . -type f -name '*.gz' | wc -l`" -gt 0 ]; do find -type f -name "*.gz" -exec gunzip '{}' \;; done
