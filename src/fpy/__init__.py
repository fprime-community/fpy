# Set Decimal precision to 30 for consistent float arithmetic across the codebase.
# This must be done at module load time to ensure all Decimal operations use this precision.
from decimal import getcontext

getcontext().prec = 30
