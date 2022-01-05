import fs from 'fs'

export type Domain = { 
    authorId: string,
    desc: string,
    script: string
}

export type DomainsDB = {
    [key: string]: Domain
}

export class Domains {
    private domains: DomainsDB
    private path: string

    private flush(): Promise<void> {
        return new Promise((resolve, reject) => {
            fs.writeFile(this.path, JSON.stringify(this.domains), (err) => {
                if (err) throw reject(err)
                resolve()
            })
        })
    }

    constructor(path: string) {
        this.path = path
        try {
            this.domains = JSON.parse(fs.readFileSync(path).toString()) as DomainsDB
        } catch (err) {
            throw new Error(`Error when initializing Domains DB: ${err}`)
        }
    }

    getAll(): [string, Domain][] {
        return Object.entries(this.domains)
    }

    get(name: string): Domain | undefined {
        return this.domains[name]
    }

    ownsDomain(domain: string, authorId: string): boolean {
        return this.domains[domain]?.authorId == authorId
    }

    create(domain: string, desc: string, authorId: string, script: string) {
        this.domains[domain] = {
            authorId,
            script,
            desc
        }

        this.flush()
    }

    updateDomain(domain: string, script: string, desc?: string) {
        this.domains[domain].script = script

        if (desc) {
            this.domains[domain].desc = desc
        }

        this.flush()
    }
}