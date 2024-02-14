trap 'catch_error' ERR

catch_error() {
    echo "Halted due to error!"
    exit 1
}

