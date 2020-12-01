#!/bin/bash

if [ -z "$1" ] || [ "$1" = "-h" ]
  then
    echo -e 'Grab the "Future Releases" section from the release notes and format it in markdown for GitHub releases

Usage: tools/format_release_notes.sh 0.16.1'
    exit 0
fi
new_evalml_version=$1
content_raw=$(cat docs/source/release_notes.rst |  sed -n -e '/^\*\*Future Releases\*\*/,/^\*\*v/p;/^\*\*v/q' | sed '/^\.\. warning::$/d' | tail -n +2 | sed '$ d' | awk 'NF')
date_formatted=$(date '+%b. %-d, %Y')
content_markdown=$(echo -n "${content_raw}" | sed 's/^        \* /- /g' | sed 's/^    \* /### /g' | sed 's/    \*\*Breaking Changes\*\*/### Breaking Changes/g' | sed -E 's/:pr:\`([0-9]+)\`/#\1/g')
full_markdown=$(echo -e "# v${new_evalml_version} ${date_formatted}\n${content_markdown}")
echo -e "${full_markdown}"
