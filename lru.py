from collections import OrderedDict
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            print(Fore.RED + f"❌ Cache Miss for {key}")
            return -1
        else:
            self.cache.move_to_end(key)
            print(Fore.GREEN + f"✅ Cache Hit for {key}")
            return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            removed_key, removed_value = self.cache.popitem(last=False)
            print(Fore.YELLOW + f"⚠️ Cache Full. Removing least recently used item: {removed_key}")

    def display(self):
        print(Fore.CYAN + "\n📦 Current Cache State:")
        print(Fore.CYAN + "-"*30)
        for key, value in self.cache.items():
            print(Fore.CYAN + f"| Key: {key} | Value: {value} |")
        print(Fore.CYAN + "-"*30)

if __name__ == "__main__":
    print(Fore.MAGENTA + "\n🚀 Welcome to LRU Cache Memory Simulator 🚀\n")

    cap = int(input("Enter Cache Capacity: "))
    lru = LRUCache(cap)
    
    while True:
        print(Fore.BLUE + "\nOptions:")
        print(Fore.YELLOW + "1️⃣  Put (Insert/Update)")
        print(Fore.YELLOW + "2️⃣  Get (Access)")
        print(Fore.YELLOW + "3️⃣  Display Cache")
        print(Fore.YELLOW + "4️⃣  Exit")
        
        try:
            choice = int(input("\nEnter your choice: "))
        except ValueError:
            print(Fore.RED + "Invalid input! Please enter a number.")
            continue

        if choice == 1:
            key = int(input("Enter Key: "))
            value = int(input("Enter Value: "))
            lru.put(key, value)
        elif choice == 2:
            key = int(input("Enter Key to Access: "))
            lru.get(key)
        elif choice == 3:
            lru.display()
        elif choice == 4:
            print(Fore.MAGENTA + "👋 Exiting... Thanks for using the simulator!")
            break
        else:
            print(Fore.RED + "❗ Invalid Choice! Please try again.")