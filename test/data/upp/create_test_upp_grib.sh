
# Extract a subset of messages from a GRIB2 file
#
# To see all possible messages, run `wgrib2 -v <file name>`

module load wgrib2

# Messages:
#   620 - Surface pressure
#   533, 520, 506, 492 - Temperature at 1000 - 925 hPa
#   628 - 2-m temperature
#   564 - Composite reflectvity

all_msg=( 620 533 520 506 492 628 564 )
nmem=3
fnames=( "rrfs.t15z.prslev.f000.subconus.grib2"
         "rrfs.t15z.prslev.f001.subconus.grib2" )

# Create list of messages for wgrib2 command
str='^('
for m in ${all_msg[@]}; do
  str="${str}|${m}"
done
str="${str}):"

# Execute wgrib2 command
for n in $(seq 1 ${nmem}); do
  npad=$(printf "%03d" ${n})
  cd "mem${npad}"
  for f in ${fnames[@]}; do
    wgrib2 ${f} -match "${str}" -grib TEST_${f}
  done
  cd ..
done
