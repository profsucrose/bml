import { App, LogLevel } from '@slack/bolt'
import axios, { AxiosResponse } from 'axios'
import * as fs from 'fs'
import { spawn } from 'child_process'
import { unescape } from 'html-escaper'
import { ChatPostEphemeralResponse, ChatPostMessageResponse, FilesUploadResponse } from '@slack/web-api'
import { Domains } from './Domains'
import { Domain } from 'domain'
import { stringify } from 'querystring'

const { SC_TOKEN, SLACK_TOKEN, SLACK_SIGNING_SECRET } = JSON.parse(fs.readFileSync('auth.json').toString())

const app = new App({
    token: SLACK_TOKEN,
    signingSecret: SLACK_SIGNING_SECRET,
    logLevel: LogLevel.DEBUG
})

const BOT_ID = 'U02RMPDLXDH'
const CHANNEL_ID = 'C02RJRTK8LV'
const TMP_IMAGE_DOWNLOAD = './tmp/tmp.png'
const TMP_SCRIPT_DOWNLOAD = './tmp/script.bml'

const PORT = 3004

const domains = new Domains('./domains.json')

enum Cmd {
    Eval = 'eval',
    Process = 'process',
    New = 'new'
}

type Args = {
    cmd: Cmd,
    width: number,
    height: number,
    frameCount: number,
    desc?: string,
    domain?: string
}

/**
 * Download given image from URL to local filesystem 
 * @param url URI for image to download
 * @param path Path to write downloaded image to
 */
async function downloadImage(url: string, path: string) {
    const response = await axios({
        url,
        method: 'get',
        responseType: 'stream',
        headers: {
            Authorization: `Bearer ${SLACK_TOKEN}`
        }
    })

    response.data.pipe(fs.createWriteStream(path))
}

/**
 * Run BML binary with given arguments
 * @param args Command line arguments for BML
 * @returns TODO: generator for each line of stdout
 */
async function runBML(args: string[]): Promise<string> {
    const bml = spawn('../bml/target/release/bml', args)

    return new Promise((resolve, reject) => {
        let output = ''

        bml.stdout.on('data', (data: Buffer) => { output += data.toString() })
        bml.stderr.on('data', (data: Buffer) => { output += data.toString() })

        bml.on('exit', (code, signal) => {
            if (code == 0) resolve(output)
            reject(output)
        })
    })
}

/**
 * Fetches last image (PNG) in BML Slack channel
 * @returns Promise of the URL for the last sent image
 */
async function lastImage(): Promise<string> {
    let history = await app.client.conversations.history({
        channel: CHANNEL_ID
    })

    for (; ;) {
        for (let msg of history.messages || []) {
            if (msg.files && msg.files[0].filetype == 'png') {
                return msg.files[0].url_private_download!
            }
        }

        if (history.response_metadata?.next_cursor) {
            history = await app.client.conversations.history({
                channel: CHANNEL_ID,
                cursor: history.response_metadata.next_cursor!
            })
            continue
        }

        // no more pages left
        break
    }

    throw new Error('couldn\'t find image in channel history')
}

async function writeFile(path: string, text: string): Promise<void> {
    return new Promise((resolve, reject) => {
        fs.writeFile(path, text, (err: any) => {
            if (err) reject(err)
            resolve()
        })
    })
}

async function reply(text: string, threadTs: string): Promise<ChatPostMessageResponse> {
    return app.client.chat.postMessage({
        channel: CHANNEL_ID,
        text,
        thread_ts: threadTs
    })
}

async function sayEphemeral(text: string, user: string): Promise<ChatPostEphemeralResponse> {
    return app.client.chat.postEphemeral({
        channel: CHANNEL_ID,
        text,
        user
    })
}

async function replyWithImage(path: string, threadTs: string): Promise<FilesUploadResponse> {
    return app.client.files.upload({
        channels: CHANNEL_ID,
        initial_comment: 'Manipulated image',
        file: fs.createReadStream(path),
        thread_ts: threadTs
    })
}

async function scPay(receiverId: string, cents: number, forMessage: string): Promise<AxiosResponse<any, any>> {
    return axios.post('https://misguided.enterprises/scales/api/pay', {
        receiverId: receiverId,
        cents: cents,
        for: forMessage
    })
}

function parseArgs(args: string): Args {
    let result: Args = {
        cmd: null,
        width: 100,
        height: 100,
        frameCount: 1,
        desc: null,
        domain: null
    }

    const cmd = args.split(' ')[0]

    switch (cmd) {
        case 'process':
        case '':
            result.cmd = Cmd.Process
            break
        case 'eval':
            result.cmd = Cmd.Eval
            break
        case 'new':
            result.cmd = Cmd.New
            break
        default:
            throw new Error(`Expected no command specified/'process', 'eval', or 'new', got '${cmd}'`)
    }

    const flags = args.match(/\w+?=\w+/g)

    for (const flag of flags || []) {
        const parts = flag.split('=')

        const name = parts[0]

        const value = parts[1]

        const valueAsNum = (key: string) => {
            const v = parseInt(value)

            if (isNaN(v)) {
                throw new Error(`Expected number for '${key}' flag, got '${value}'`)
            }

            return v
        }

        switch (name) {
            case 'd':
            case 'domain':
                result.domain = value
                break
            case 'w':
            case 'width':
                result.width = valueAsNum('width')
                break
            case 'h':
            case 'height':
                result.height = valueAsNum('height')
                break
            case 'f':
            case 'frame_count':
                result.frameCount = valueAsNum('frame_count')
                break
            case 'desc':
                result.desc = value
                break
        }
    }

    return result
}

/**
 * Helper for unescaping HTML escape codes and links
 * (frag.xyz -> <http://frag.xyz|frag.xyz>) when parsing
 * user scripts from Bolt
 * @param text Text to be escaped
 * @returns Escaped text
 */
function unescapeSlack(text: string): string {
    return unescape(text.replaceAll(/<http\:\/\/([^|]+)\|(.+)>/g, '$1'))
}

app.command('/domain', async ({ command, ack, respond }) => {
    await ack()

    const name = command.text.trim()

    if (name == '') {
        respond('Expected /domain <name>')
        return
    }

    const domain = domains.get(name)

    if (domain == undefined) {
        respond(`No-one currently owns the domain '${name}' (but you could buy your own for just *20 sc: sc!* (or, for now, for free with /buydomain <name>!))`)
        return
    }

    const message = `\`${name}\`: ${domain.desc}\n${
        domain.script == '' 
            ? '<No script currently exists>'
            : `\`\`\`${domain.script}\`\`\``
    }`

    await respond(message)
})

app.command('/domains', async ({ ack, respond }) => {
    await ack()

    const list = domains.getAll()

    const message = list.length 
        ? list.map(([name, dom]) => `${name}: owned by <@${dom.authorId}>, "${dom.desc}"`).join('\n')
        : "Currently no-one owns any domains! (but you could buy for one for only *20 :sc: big ones!* (or for now just with /buydomain <name> for free)"

    await respond(message)
})

app.command('/buydomain', async ({ command, ack, respond }) => {
    await ack()

    const { user_id, text } = command

    // /buy <domain>
    const parts = text.match(/(\w+) ["“](.*)["”]/)

    console.log(text, parts)

    if (parts?.length < 3) {
        respond('Expected /buy <domain> "What this script does"')
        return
    }

    const name = parts[1].trim()

    const domain = domains.get(name)

    if (domain?.authorId == user_id) {
        respond('You already own this domain!')
        return
    }

    const desc = parts[2].trim()

    try {
        // const result = scPay(user_id, 20);
        await respond(`You now own the script domain ${name}!`)

        domains.create(name, desc, user_id, '')
    } catch (error: any) {
        // respond(`You need at least **:sc: 20** to buy the domain ${text}, but only have ${error.balance} in your balance`)
    }
})

app.message(async ({ message }) => {
    const text = (message as any).text as string

    console.log('received message:', message)

    const ts = (message as any).thread_ts || message.ts
    const authorId = (message as any).user

    const messageRe = new RegExp(`<@${BOT_ID}>([^\`]+)(?:\`\`\`([^\`]*)\`\`\`)?`)
    const groups = text.match(messageRe)

    // return if doesn't match
    if (!groups) {
        console.log('Returning')
        return
    }

    const argsString = groups[1].trim()

    let args: Args

    try {
        args = parseArgs(argsString)
    } catch (error: any) {
        await sayEphemeral(error.message, authorId)
        return
    }

    console.log(groups)

    let script = groups[2] ? unescapeSlack(groups[2].trim()) : null

    console.log('SCRIPT', script)

    if (script == undefined || script == '') {
        if (args.domain == null) {
            await sayEphemeral('Expected non-empty script code block or domain (d=<domian> or domain=<domain>), got neither', authorId)
            return
        }

        const domain = domains.get(args.domain)

        if (domain != null) {
            if (domain.script == '') {
                await reply(`'${args.domain}' is owned by <@${domain.authorId}>, but currently has no assigned script!`, authorId)
                return
            }

            script = domain.script
        } else {
            await sayEphemeral(`No-one owns the domain '${args.domain}' (but you could get it for just **20 :sc:** :eyes:)`, authorId)
            return
        }
    }

    writeFile(TMP_SCRIPT_DOWNLOAD, script)

    console.log(`HANDLING COMMAND: ${args.cmd}`)
    switch (args.cmd) {
        case Cmd.Eval: {
            try {
                const result = await runBML(['eval', TMP_SCRIPT_DOWNLOAD])
                reply(`\`\`\`${result}\`\`\``, ts)
            } catch (err) {
                reply(`Error: \`\`\`${err}\`\`\``, ts)
            }

            break
        }

        case Cmd.Process: {
            const url = await lastImage()

            await downloadImage(url, TMP_IMAGE_DOWNLOAD)

            try {
                const output = `${ts}.${args.frameCount > 1 ? 'gif' : 'png'}`
                await runBML(['process', TMP_SCRIPT_DOWNLOAD, TMP_IMAGE_DOWNLOAD, args.frameCount.toString(), output])
                replyWithImage(output, ts)
            } catch (err) {
                reply(`Error: \`\`\`${err}\`\`\``, ts)
            }

            break
        }

        case Cmd.New: {
            try {
                const output = `${ts}.${args.frameCount > 1 ? 'gif' : 'png'}`
                await runBML(['new', TMP_SCRIPT_DOWNLOAD, args.width.toString(), args.height.toString(), args.frameCount.toString(), output])
                replyWithImage(output, ts)
            } catch (err) {
                reply(`Error: \`\`\`${err}\`\`\``, ts)
                return
            }

            break
        }
    }

    // if new script was given and domain specified
    if (args.domain != null && script != null) {
        if (!domains.ownsDomain(args.domain, authorId)) {
            reply(`listen here whippersnapper, you don't even own the domain "${args.domain}" it is, however, available. if you want to buy it for 20:sc:, do /bml buy ${args.domain}`, ts)
            return
        }

        // reply(`we've associated it with your "${args.domain}" domain. now anyone can run <@${BOT_ID}> ${args.domain} anywhere in the slack, and I will grab the last image someone posted and "${args.domain}" it!`, ts)

        domains.updateDomain(args.domain, script)

        reply(`Updated script on domain ${args.domain}`, ts)
    }
})

;(async () => {
    await app.start(PORT)
    console.log(`The Slack app has started!`)
})()