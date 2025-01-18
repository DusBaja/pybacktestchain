import hashlib
import time
from dataclasses import dataclass, field
import pickle # prefered serialization method

@dataclass
class Block:
    name_backtest: str
    data: str
    previous_hash: str = ''
    timestamp: float = field(default_factory=time.time)
    hash: str = field(init=False)

    def __post_init__(self):
        # Automatically calculate the hash after initialization
        self.hash = self.calculate_hash

    @property
    def calculate_hash(self):
        return hashlib.sha256(
            (str(self.timestamp) 
             + self.name_backtest
             + self.data 
             + self.previous_hash).encode()
        ).hexdigest()
    
@dataclass
class Blockchain:
    name: str
    chain: list = field(default_factory=list)

    def store(self):
        with open(f'blockchain/{self.name}.pkl', 'wb') as f:
            pickle.dump(self, f)

    def __post_init__(self):
        # Initialize the chain with the genesis block
        self.chain.append(self.create_genesis_block())
        self.store()
        

    def create_genesis_block(self):
        return Block('Genesis Block', '', '0')

    def add_block(self, name:str, data: str):
        previous_block = self.chain[-1]
        new_block = Block(name, data, previous_block.hash)
        self.chain.append(new_block)
        self.store()

    def is_valid(self):
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]

            if current_block.hash != current_block.calculate_hash:
                return False

            if current_block.previous_hash != previous_block.hash:
                return False

        return True
    
    def __str__(self):
        # display the blockchain
        to_return = ''
        for i, block in enumerate(self.chain):
            to_return += "-" * 80 + '\n'
            to_return += f"Block {i}\n"
            to_return += "-" * 80 + '\n'
            to_return += f"Backtest: {block.name_backtest}\n"
            to_return += f"Timestamp: {block.timestamp}\n"
            to_return += f"Hash: {block.hash}\n"
            to_return += f"Previous Hash: {block.previous_hash}\n"
            to_return += "-" * 80 + '\n'
        return to_return
    

def load_blockchain(name: str):
    with open(f'blockchain/{name}.pkl', 'rb') as f:
        return pickle.load(f)