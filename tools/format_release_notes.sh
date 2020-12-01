#!/bin/bash

if [ "$1" = "-h" ]
  then
    echo -e 'Grab the "Future Releases" section from the release notes and format it in markdown for GitHub releases

Usage: tools/format_release_notes.sh'
    exit 0
fi
evalml_version=$(cat setup.py | sed -n -E "s/.*version=\'([0-9A-Za-z\.\_]+)\'.*/\1/p")
#content_raw=$(cat docs/source/release_notes.rst |  sed -e "/^\*\*v${evalml_version}[A-Za-z0-9\.\,\ ]+\*\*$/,/^\*\*v/p;/^\*\*v/q" | sed '/^\.\. warning::$/d' | tail -n +2 | sed '$ d' | awk 'NF')
content_raw=$(cat docs/source/release_notes.rst |  sed -n -e "/^\*\*v0.16.1/,/^\*\*v0.16.0/p;/^\*\*v0.16.0/q" | sed '/^\.\. warning::$/d' | tail -n +2 | sed '$ d' | awk 'NF')
date_formatted=$(date '+%b. %-d, %Y')
content_markdown=$(echo -n "${content_raw}" | sed 's/^        \* /- /g' | sed 's/^    \* /### /g' | sed 's/    \*\*Breaking Changes\*\*/### Breaking Changes/g' | sed -E 's/:pr:\`([0-9]+)\`/#\1/g')
full_markdown=$(echo -e "# v${evalml_version} ${date_formatted}\n${content_markdown}")
echo -e "${full_markdown}"
