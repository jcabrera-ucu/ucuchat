import dip from 'dumpster-dip'

const opts = {
  input: 'eswiki.xml',
  outputMode: 'encyclopedia',
  parse: function(doc) {
    return doc.sentences().map(x => x.text()).join('\n')
  }
}

// this promise takes ~4hrs
dip(opts).then(() => {
  console.log('done!')
})
