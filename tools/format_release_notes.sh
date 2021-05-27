#!/bin/bash

if [ "$1" = "-h" ]
  then
    echo -e 'Grab the "Future Releases" section from the release notes and format it in markdown for GitHub releases

Usage: tools/format_release_notes.sh

Note: this was written to work properly on mac. It should work on linux if -r is used instead of -E for sed.'
    exit 0
fi
evalml_version=$(sed -n -E "s/.*version=\'([0-9A-Za-z\.\_]+)\'.*/\1/p" setup.py)
content_raw=$(sed -n -E '/^\*\*v[A-Za-z0-9\., ]+\*\*$/,$p' docs/source/release_notes.rst | tail -n +2 | sed -E '/\*\*v[A-Za-z0-9\., ]+\*\*$/q' | sed '$ d' | awk 'NF')
date_formatted=$(date '+%b. %-d, %Y')
content_markdown=$(echo -n "${content_raw}" | grep -v ".. warning::" | sed 's/^        \* /- /g' | sed 's/^    \* /### /g' | sed 's/    \*\*Breaking Changes\*\*/### Breaking Changes/g' | sed -E "s/:pr:\`([0-9]+)\`/#\1/g")
full_markdown=$(echo -e "# v${evalml_version} ${date_formatted}\n${content_markdown}")
echo -e "${full_markdown}"
