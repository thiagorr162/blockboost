echo "FIELDS"
head -n1 "$1" | awk -F, '{for (i=1; i<=NF-1; i++) print i, $i }' | head -n-1
echo
echo "UNIQUE MISSING PATTERNS"
echo '...'
tail -n+2 "$1" | awk -F, '{for (i=1; i<=NF-1; i++) if( $i == "") printf 0 ; else printf 1 ; print substr($(NF), 4,1) }' | sort | uniq -c | sort -g | tail -n40
echo
echo "NUMBER OF MISSING BY TYPE AND FIELD NUMBER"
echo '...'
tail -n+2 "$1" | awk -F, '{for (i=1; i<=NF-1; i++) if($i == "") m[substr($(NF), 4,1) i] += 1 } END { for (key in m) print m[key], key}' | grep A |sort -g | tail -n20
echo '...'
tail -n+2 "$1" | awk -F, '{for (i=1; i<=NF-1; i++) if($i == "") m[substr($(NF), 4,1) i] += 1 } END { for (key in m) print m[key], key}' |grep B | sort -g | tail -n20
echo
echo "TOTAL BY TYPE"
tail -n+2 "$1" | awk -F, '{m[substr($(NF), 4,1)] += 1 } END { for (key in m) print m[key] " " key}' | sort -g
echo
echo

