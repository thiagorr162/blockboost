BEGIN {
    FS=",";
    lut[1] = "A";
    lut[2] = "B";
    lut[3] = "C";
    lut[4] = "D";
    lut[5] = "E";
};

NR == 1 {
    for (i=6;i<NF; i++) {
        printf "%s,", $i;
    }
    print "entity_id,record_id";
};

NR != 1 {
    for (i=6;i<NF; i++) {
        printf "%s,", $i;
    }

    printf $2 ",";
    printf "id_" lut[$4] "-" $1 "\n";
};

