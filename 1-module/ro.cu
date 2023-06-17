#include <cryptoTools/Crypto/random_oracle.h>

namespace osuCrypto
{
	const u64 Blake2::HashSize;
	const u64 Blake2::MaxHashSize;

    const Blake2& Blake2::operator=(const Blake2& src)
    {
        state = src.state;
        return *this;
    }
}
