# Ethereum Notes

Address: basic hexadecimal ethereum format
Balance: integer, 0 to 10^75

Contracts immutable once deployed
Contracts can inherit from (more abstract) contracts (or contract components?)

Token can be **owned**, owner can be a user (a wallet?) or a **contract**.

Contracts can be programmed to function as administrators, given input from token owners.

Token Ownership == Group Membership



# Possible Sources

The Modern Utopian
The Ascent of Money
Prototype Theory Book
Dunbar's Number

# Ideas

Create contract component?

Recursively decompose contracts into subcontracts of ~150 people most closely aligned on a particular issue?

# Gas

1 Gas = 1 Unit of computation

# Contracts

http://www.ethdocs.org/en/latest/contracts-and-transactions/account-types-gas-and-transactions.html

Contracts generally serve four purposes:

Maintain a data store representing something which is useful to either other contracts or to the outside world; one example of this is a contract that simulates a currency, and another is a contract that records membership in a particular organization.
Serve as a sort of externally-owned account with a more complicated access policy; this is called a “forwarding contract” and typically involves simply resending incoming messages to some desired destination only if certain conditions are met; for example, one can have a forwarding contract that waits until two out of a given three private keys have confirmed a particular message before resending it (ie. multisig). More complex forwarding contracts have different conditions based on the nature of the message sent. The simplest use case for this functionality is a withdrawal limit that is overrideable via some more complicated access procedure. A wallet contract is a good example of this.
Manage an ongoing contract or relationship between multiple users. Examples of this include a financial contract, an escrow with some particular set of mediators, or some kind of insurance. One can also have an open contract that one party leaves open for any other party to engage with at any time; one example of this is a contract that automatically pays a bounty to whoever submits a valid solution to some mathematical problem, or proves that it is providing some computational resource.
Provide functions to other contracts, essentially serving as a software library.