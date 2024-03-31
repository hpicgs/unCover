import os
import tensorflow as tf
import numpy as np
import json

from grover.lm.modeling import GroverConfig, sample
from grover.sample.encoder import get_encoder, _tokenize_article_pieces, extract_generated_target
from grover.download_model import download_grover
from tqdm import tqdm

# currently can only process one document at a time in comparison to original implementation
def generate_grover_news_from_original(doc, model_type, model_dir):
    print("Starting Grover Processing")
    encoder = get_encoder()
    grover_path = os.path.join(model_dir, "grover-" + model_type)
    if not os.path.exists(grover_path):
        download_grover(model_type, model_dir)
    news_config = GroverConfig.from_json_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs",
                                                           model_type + ".json"))

    # We might have to split the batch into multiple chunks if the batch size is too large
    default_mbs = {12: 32, 24: 16, 48: 3}
    max_batch_size = default_mbs[news_config.num_hidden_layers]

    # factorize batch_size (1) = (num_chunks * batch_size_per_chunk) s.t. batch_size_per_chunk < max_batch_size
    num_chunks = int(np.ceil(1 / max_batch_size))
    batch_size_per_chunk = int(np.ceil(1 / num_chunks))
    print("\n~~\nbatch size={}, max batch size={}, num chunks={}, batch size per chunk={}\n~~\n".format(
        1, max_batch_size, num_chunks, batch_size_per_chunk), flush=True)

    # This controls the top p for each generation.
    top_p = np.ones((num_chunks, batch_size_per_chunk), dtype=np.float32) * 0.95

    articles = [json.loads(doc)]

    tf_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)

    with tf.compat.v1.Session(config=tf_config, graph=tf.Graph()) as sess:
        initial_context = tf.compat.v1.placeholder(tf.int32, [batch_size_per_chunk, None])
        p_for_topp = tf.compat.v1.placeholder(tf.float32, [batch_size_per_chunk])
        eos_token = tf.compat.v1.placeholder(tf.int32, [])
        ignore_ids = tf.compat.v1.placeholder(tf.bool, [news_config.vocab_size])
        tokens, probs = sample(news_config=news_config, initial_context=initial_context,
                               eos_token=eos_token, ignore_ids=ignore_ids, p_for_topp=p_for_topp,
                               do_topk=False)

        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, os.path.join(grover_path, "model.ckpt"))

        # Let's go!
        for i, article in enumerate(tqdm(articles)):
            article_pieces = _tokenize_article_pieces(encoder, article)
            context_formatted = []
            for key in ['domain', 'date', 'authors', 'title', 'article']:
                if key != 'article':
                    context_formatted.extend(article_pieces.pop(key, []))

            if len(context_formatted) >= 1020:
                print(
                    "WARNING: the provided context is {} tokens, but the maximum length Grover was trained on was 1024 tokens.".format(
                        len(context_formatted)), flush=True)
                context_formatted = context_formatted[:1020]

            context_formatted.append(encoder.__dict__['begin_{}'.format('article')])
            # Format context end

            # Indices we definitely DONT WANT TO PREDICT
            ignore_ids_np = np.array(encoder.special_tokens_onehot)
            #ignore_ids_np[encoder.__dict__['end_{}'.format('target')]] = 0

            gens = []
            gens_raw = []
            gen_probs = []

            article['top_ps'] = top_p.reshape(-1).tolist()
            for chunk_i in range(num_chunks):
                tokens_out, probs_out = sess.run([tokens, probs],
                                                 feed_dict={initial_context: [context_formatted] * batch_size_per_chunk,
                                                            eos_token: encoder.__dict__['end_{}'.format('article')],
                                                            ignore_ids: ignore_ids_np,
                                                            p_for_topp: top_p[chunk_i]})

                for t_i, p_i in zip(tokens_out, probs_out):
                    extraction = extract_generated_target(output_tokens=t_i, encoder=encoder, target='article')
                    gens.append(extraction['extraction'])

                    # NOTE: Originally I didn't add the +1 which meant that end article was being cut off. whoops.
                    # better add that!
                    gens_raw.append(t_i[extraction['start_ind']:extraction['end_ind'] + 1].tolist())

                    assert extraction['start_ind'] == len(context_formatted)
                    gen_probs.append(p_i[:extraction['end_ind'] - len(context_formatted) + 1].tolist())

            article['gens_{}'.format('article')] = gens
            article['gensraw_{}'.format('article')] = gens_raw
            article['probs_{}'.format('article')] = gen_probs

            # these were in there for whatever reason...
            article.pop('input_ids_conditional', None)
            article.pop('input_ids_unconditional', None)
            print("Written {}/{} articles. Finished Grover".format(i+1, len(articles)), flush=True)
            return article["gens_article"][0]
