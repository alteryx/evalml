allow_list=$(cat core-requirements.txt requirements.txt | grep -oE "^[a-zA-Z0-9]+[a-zA-Z0-9_\-]*" | paste -d "|" -s -)
cmd_builder_pkg="cmdstan-builder"
allow_list="${allow_list}|${cmd_builder_pkg}"
echo "Allow list: ${allow_list}"
pip freeze | grep -v "evalml.git" | grep -E "${allow_list}" > "${DEPENDENCY_FILE_PATH}"
