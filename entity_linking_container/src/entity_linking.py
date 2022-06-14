import os
import ast
import time
import pathlib
import argparse
import gc
import blink.ner as NER
import blink.main_dense as main_dense

from blink.crossencoder.train_cross import modify
from blink.crossencoder.data_process import prepare_crossencoder_data


def run(
    args,
    logger,
    biencoder,
    biencoder_params,
    crossencoder,
    crossencoder_params,
    candidate_encoding,
    title2id,
    id2title,
    id2text,
    wikipedia_id2local_id,
    faiss_indexer=None,
    test_data=None,
):

    if not test_data and not args.test_mentions and not args.interactive:
        msg = (
            "ERROR: either you start BLINK with the "
            "interactive option (-i) or you pass in input test mentions (--test_mentions)"
            "and test entitied (--test_entities)"
        )
        raise ValueError(msg)

    id2url = {
        v: "https://en.wikipedia.org/wiki?curid=%s" % k
        for k, v in wikipedia_id2local_id.items()
    }

    stopping_condition = False
    while not stopping_condition:
        samples = None

        if args.interactive:
            logger.info("interactive mode")

            # biencoder_params["eval_batch_size"] = 1

            # Load NER model
            ner_model = NER.get_model()

            # Interactive
            text = input("insert text:")

            # Identify mentions
            samples = main_dense._annotate(ner_model, [text])

            main_dense._print_colorful_text(text, samples)

        else:
            if logger:
                logger.info("test dataset mode")

            if test_data:
                samples = test_data
            else:
                # Load test mentions
                samples = main_dense._get_test_samples(
                    args.test_mentions,
                    args.test_entities,
                    title2id,
                    wikipedia_id2local_id,
                    logger,
                )

            stopping_condition = True

        # don't look at labels
        keep_all = (
            args.interactive
            or samples[0]["label"] == "unknown"
            or samples[0]["label_id"] < 0
        )

        # prepare the data for biencoder
        if logger:
            logger.info("preparing data for biencoder")
        dataloader = main_dense._process_biencoder_dataloader(
            samples, biencoder.tokenizer, biencoder_params
        )

        # run biencoder
        if logger:
            logger.info("run biencoder")
        top_k = args.top_k

        labels, nns, scores = main_dense._run_biencoder(
            biencoder, dataloader, candidate_encoding, top_k, faiss_indexer
        )

        if args.interactive:

            print("\nfast (biencoder) predictions:")

            main_dense._print_colorful_text(text, samples)

            # print biencoder prediction
            idx = 0
            for entity_list, sample in zip(nns, samples):
                e_id = entity_list[0]
                e_title = id2title[e_id]
                e_text = id2text[e_id]
                e_url = id2url[e_id]
                main_dense._print_colorful_prediction(
                    idx, sample, e_id, e_title, e_text, e_url, args.show_url
                )
                idx += 1
            print()

            if args.fast:
                # use only biencoder
                continue

        else:

            biencoder_accuracy = -1
            recall_at = -1
            if not keep_all:
                # get recall values
                top_k = args.top_k
                x = []
                y = []
                for i in range(1, top_k):
                    temp_y = 0.0
                    for label, top in zip(labels, nns):
                        if label in top[:i]:
                            temp_y += 1
                    if len(labels) > 0:
                        temp_y /= len(labels)
                    x.append(i)
                    y.append(temp_y)
                # plt.plot(x, y)
                biencoder_accuracy = y[0]
                recall_at = y[-1]
                print("biencoder accuracy: %.4f" % biencoder_accuracy)
                print("biencoder recall@%d: %.4f" % (top_k, y[-1]))

            if args.fast:

                predictions = []
                for entity_list in nns:
                    sample_prediction = []
                    for e_id in entity_list:
                        e_title = id2title[e_id]
                        sample_prediction.append(e_title)
                    predictions.append(sample_prediction)

                # use only biencoder
                return (
                    biencoder_accuracy,
                    recall_at,
                    -1,
                    -1,
                    len(samples),
                    predictions,
                    scores,
                )

        # prepare crossencoder data
        context_input, candidate_input, label_input = prepare_crossencoder_data(
            crossencoder.tokenizer,
            samples,
            labels,
            nns,
            id2title,
            id2text,
            keep_all,
        )

        context_input = modify(
            context_input, candidate_input, crossencoder_params["max_seq_length"]
        )

        dataloader = main_dense._process_crossencoder_dataloader(
            context_input, label_input, crossencoder_params
        )

        # run crossencoder and get accuracy
        accuracy, index_array, unsorted_scores = main_dense._run_crossencoder(
            crossencoder,
            dataloader,
            logger,
            context_len=biencoder_params["max_context_length"],
        )

        if args.interactive:

            print("\naccurate (crossencoder) predictions:")

            main_dense._print_colorful_text(text, samples)

            # print crossencoder prediction
            idx = 0
            for entity_list, index_list, sample in zip(nns, index_array, samples):
                e_id = entity_list[index_list[-1]]
                e_title = id2title[e_id]
                e_text = id2text[e_id]
                e_url = id2url[e_id]
                main_dense._print_colorful_prediction(
                    idx, sample, e_id, e_title, e_text, e_url, args.show_url
                )
                idx += 1
            print()
        else:

            scores = []
            predictions = []
            links = []
            for entity_list, index_list, scores_list in zip(
                nns, index_array, unsorted_scores
            ):

                index_list = index_list.tolist()

                # descending order
                index_list.reverse()

                sample_prediction = []
                sample_links = []  # added
                sample_scores = []
                for index in index_list:
                    e_id = entity_list[index]
                    e_title = id2title[e_id]
                    e_url = id2url[e_id]  # added
                    sample_prediction.append(e_title)
                    sample_links.append(e_url)  # added
                    sample_scores.append(scores_list[index])
                predictions.append(sample_prediction)
                scores.append(sample_scores)
                links.append(sample_links)  # added

            crossencoder_normalized_accuracy = -1
            overall_unormalized_accuracy = -1
            if not keep_all:
                crossencoder_normalized_accuracy = accuracy
                print(
                    "crossencoder normalized accuracy: %.4f"
                    % crossencoder_normalized_accuracy
                )

                if len(samples) > 0:
                    overall_unormalized_accuracy = (
                        crossencoder_normalized_accuracy
                        * len(label_input)
                        / len(samples)
                    )
                print(
                    "overall unnormalized accuracy: %.4f" % overall_unormalized_accuracy
                )
            return (
                biencoder_accuracy,
                recall_at,
                crossencoder_normalized_accuracy,
                overall_unormalized_accuracy,
                len(samples),
                predictions,
                links,  # added
                scores,
            )


def get_entities(data_to_link, args, models):
    data_to_link = ast.literal_eval(data_to_link)

    for data in data_to_link:

        if len(data["context_left"].split()) > 100:
            data["context_left"] = data["context_left"].split()[-100:]
            data["context_left"] = " ".join(data["context_left"])

        if len(data["context_right"].split()) > 100:
            data["context_right"] = data["context_right"].split()[:100]
            data["context_right"] = " ".join(data["context_right"])

        if len(data["mention"].split()) > 100:
            print("Mention longer than 100 tokens: ", data["mention"])

    if len(data_to_link) > 0:
        try:
            _, _, _, _, _, predictions, links, scores = run(
                args, None, *models, test_data=data_to_link
            )
            error = False
        except Exception as e:
            print(e)
            print("Error while performing entity linking on data row")
            error = True
    else:
        error = True

    entities_dict = {}
    if not error:
        ent_list = []
        identified_entities = []
        for i in range(0, len(data_to_link)):
            ent_dict = {}
            print("Mention: ", data_to_link[i]["mention"])
            print("Original sentence: ")
            print(
                data_to_link[i]["context_left"]
                + " "
                + data_to_link[i]["mention"]
                + " "
                + data_to_link[i]["context_right"]
            )
            print("Entity linked: ", predictions[i][0])
            print("Score: ", scores[i][0])
            print("\n")

            if scores[i][0] > 0:

                if (
                    predictions[i][0] not in identified_entities
                ):  # and data_to_link[i]["type"]=="GPE":
                    # ent_dict['text'] = data_to_link[i]["original sentence"]
                    ent_dict["text"] = (
                        data_to_link[i]["context_left"]
                        + " "
                        + data_to_link[i]["mention"]
                        + " "
                        + data_to_link[i]["context_right"]
                    )
                    ent_dict["type"] = data_to_link[i]["type"]
                    ent_dict["mention"] = data_to_link[i]["mention"]
                    ent_dict["entity_text"] = predictions[i][0]
                    ent_dict["entity_link"] = links[i][0]
                    ent_dict["entity_score"] = scores[i][0]
                    identified_entities.append(predictions[i][0])
            # else:
            #     ent_dict['text'] = data_to_link[i]["context_left"] + " " + data_to_link[i]["mention"] + " " + data_to_link[i]["context_right"]
            #     ent_dict['mention'] = data_to_link[i]["mention"]
            #     ent_dict['entity_text'] = ""
            #     ent_dict['entity_link'] = ""
            #     ent_dict['entity_score'] = scores[i][0]
            if ent_dict:
                ent_list.append(ent_dict)
        entities_dict["identified_entities"] = identified_entities
        entities_dict["entities"] = ent_list

    if not entities_dict:
        print("Empty dict")

    return entities_dict


# @st.cache(allow_output_mutation=True)
def entity_linking(
    df,
    args,
    model_dir,
    task_entity_col,
    task_entity_linking_col,
    fast,
    output_path,
    test_entities,
    test_mentions,
    interactive,
    top_k,
    faiss_index,
    biencoder_model,
    biencoder_config,
    entity_catalogue,
    entity_encoding,
    crossencoder_model,
    crossencoder_config,
    index_path,
):
    absolute_path = pathlib.Path(__file__).parent.resolve()

    # constructing args
    args = argparse.Namespace(**args)
    args.biencoder_model = os.path.join(absolute_path, model_dir, biencoder_model)
    args.biencoder_config = os.path.join(absolute_path, model_dir, biencoder_config)
    args.entity_catalogue = os.path.join(absolute_path, model_dir, entity_catalogue)
    args.entity_encoding = os.path.join(absolute_path, model_dir, entity_encoding)
    args.crossencoder_model = os.path.join(absolute_path, model_dir, crossencoder_model)
    args.crossencoder_config = os.path.join(
        absolute_path, model_dir, crossencoder_config
    )
    args.index_path = os.path.join(absolute_path, model_dir, index_path)

    args.fast = fast
    args.output_path = output_path
    args.test_entities = test_entities
    args.test_mentions = test_mentions
    args.interactive = interactive
    args.top_k = top_k
    args.faiss_index = faiss_index

    # # override the original run() method
    # main_dense.run = run

    # load models
    start = time.time()
    models = main_dense.load_models(args, logger=None)
    end = time.time()
    print("Time to load BLINK models", end - start)

    df[task_entity_linking_col] = df[task_entity_col].apply(
        lambda entities: get_entities(entities, args, models)
    )

    del models
    gc.collect()

    print(df.info())

    return df