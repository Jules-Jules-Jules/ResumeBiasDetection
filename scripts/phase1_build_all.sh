set -e  # Exit on error

echo "Phase 1: Data Processing Pipeline"
echo ""

cd "$(dirname "$0")/.." || exit 1

echo "Step 1/6: Cleaning resumes"
python -m src.data.clean_resumes
echo ""

echo "Step 2/6: Cleaning jobs"
python -m src.data.clean_jobs
echo ""

echo "Step 3/6: Normalizing occupations"
python -m src.data.normalize_occupations
echo ""

echo "Step 4/6: Building train/test splits"
python -m src.data.build_splits
echo ""

echo "Step 5/6: Augmenting resumes with demographic names"
python -m src.data.augment_names
echo ""

echo "Step 6/6: Building job-resume pairs"
python -m src.data.build_pairs
echo ""

echo "Phase 1 Complete"
echo ""
echo "Generated files in data/processed/:"
ls -lh data/processed/ | grep -v "^d" | awk '{print "  - " $9 " (" $5 ")"}'
echo ""
