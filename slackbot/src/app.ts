import { App, LogLevel } from '@slack/bolt'
import axios from 'axios'
import * as fs from 'fs'
import { spawn } from 'child_process'
import { unescape } from 'html-escaper'
import { ChatPostMessageResponse, FilesUploadResponse } from '@slack/web-api'

const { TOKEN, SIGNING_SECRET } = JSON.parse(fs.readFileSync('auth.json').toString())

const app = new App({
    token: TOKEN,
    signingSecret: SIGNING_SECRET,
    logLevel: LogLevel.DEBUG
})

const BOT_ID = 'U02RMPDLXDH'
const CHANNEL_ID = 'C02RJRTK8LV'
const TMP_IMAGE_DOWNLOAD = './tmp/tmp.png'
const TMP_BML_OUTPUT = './tmp/out.png'
const TMP_SCRIPT_DOWNLOAD = './tmp/script.bml'

const PORT = 3004

enum Cmd {
    Eval = 'eval',
    Process = 'process',
    New = 'new'
}

type Args = {
    cmd: Cmd,
    width: number,
    height: number,
    frameCount: number
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
            Authorization: `Bearer ${TOKEN}`
        }
    })

    response.data.pipe(fs.createWriteStream(path))
}

/**
 * Run BML binary with given arguments
 * @param args Command line arguments for BML
 * @returns TODO: generator for each line of stdout
 */
async function runBML(args: Array<string>): Promise<string> {
    const bml = spawn('../bml/target/release/bml', args)

    return new Promise((resolve, reject) => {
        let output = ''

        bml.stdout.on('data', (data: Buffer) => { output += data.toString() })
        bml.stderr.on('data', (data: Buffer) => { output += data.toString() })

        bml.on('exit', (code, signal) => {
            console.log('Process ended', code, signal)
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

    for (;;) {
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

async function replyWithImage(path: string, threadTs: string): Promise<FilesUploadResponse> {
    return app.client.files.upload({
        channels: CHANNEL_ID,
        initial_comment: 'Manipulated image',
        file: fs.createReadStream(path),
        thread_ts: threadTs
    })
}

function parseArgs(args: string): Args {
    let result: Args = {
        cmd: null,
        width: 100,
        height: 100,
        frameCount: 1
    }

    switch (args.split(' ')[0]) {
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
            throw new Error(`Expected no command specified and thus image processing by default, 'eval', or 'new', got ${cmd}`)
    }

    const flags = args.match(/([a-z_])+?=[0-9]+/g)

    for (const flag of flags || []) {
        const parts = flag.split('=')

        const name = parts[0]
        const value = parseInt(parts[1])

        switch (name) {
            case 'width':
                result.width = value
                break
            case 'height':
                result.height = value
                break
            case 'frame_count':
                result.frameCount = value
                break
        }
    }

    return result
}

app.message(async ({ message, say }) => {
    const text = (message as any).text as string

    console.log(`Text: '${text}'`)

    const ts = message.ts

    const messageRe = new RegExp(`<@${BOT_ID}>(.*?)(?:\n)?\`\`\`((.|\n)*?)\`\`\``)
    const groups = text.match(messageRe)

    console.log('Groups', groups)

    // return if not bot command
    if (!groups || groups.length < 3) {
        console.log('Returning')
        return
    }

    const argsString = groups[1].trim()

    const script = unescape(groups[2])

    const args = parseArgs(argsString)

    writeFile(TMP_SCRIPT_DOWNLOAD, script)

    switch (args.cmd) {
        case Cmd.Eval: {
            try {
                const result = await runBML(['eval', TMP_SCRIPT_DOWNLOAD])
                reply(`\`\`\`${result}\`\`\``, ts)
            } catch (err) {
                reply(`Error: \`\`\`${err}\`\`\``, ts)
            }
        }

        case Cmd.Process: {
            const url = await lastImage()

            await downloadImage(url, TMP_IMAGE_DOWNLOAD)

            try {
                await runBML(['process', TMP_SCRIPT_DOWNLOAD, TMP_IMAGE_DOWNLOAD, args.frameCount.toString(), TMP_BML_OUTPUT])

                replyWithImage(TMP_BML_OUTPUT, ts)
            } catch (err) {
                reply(`Error: \`\`\`${err}\`\`\``, ts)
            }
        }

        case Cmd.New: {
            try {
                await runBML(['new', TMP_SCRIPT_DOWNLOAD, args.width.toString(), args.height.toString(), args.frameCount.toString(), TMP_BML_OUTPUT])

                replyWithImage(TMP_BML_OUTPUT, ts)
            } catch (err) {
                reply(`Error: \`\`\`${err}\`\`\``, ts)
            }
        }
    }

    /*
    const url = await lastImage()

    console.log('DOWNLOAD URL', url)

    await downloadImage(url)

    const text = ((message as any)).text as string

    const groups = text.match(/```((.|\n)*?)```/)

    let script = groups && unescape(groups[1])

    console.log(script)

    if (!script) {
        return
    }

    console.log(`Script: '${script}'`)

    await new Promise((resolve, reject) => {
        fs.writeFile(TMP_SCRIPT_DOWNLOAD, script, (err) => {
            if (err) reject(err)
            resolve('')
        })
    })

    try {
        await runBML()

        // send images
        app.client.files.upload({
            channels: CHANNEL_ID,
            initial_comment: 'Manipulated image',
            file: fs.createReadStream(TMP_BML_OUTPUT)
        })
    } catch (err) {
        say(<string>`Error: \`\`\`${err}\`\`\``)
    }
    */
})

;(async () => {
    await app.start(PORT)
    console.log(`The Slack app has started!`)
})()