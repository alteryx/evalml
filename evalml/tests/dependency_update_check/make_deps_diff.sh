allow_list=$(cat core-requirements.txt requirements.txt | grep -oE "^[a-zA-Z0-9\-_]+" | paste -d "|" -s -)
echo $allow_list
pip freeze | grep -v "evalml.git" | grep -E ${allow_list} > ${DEPENDENCY_FILE_PATH}
