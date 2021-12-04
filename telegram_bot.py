from io import BytesIO, TextIOWrapper

import requests
import yaml

import telegram
from telegram import ReplyKeyboardMarkup
from telegram.ext import (
    Updater, 
    CommandHandler, 
    MessageHandler, 
    Filters, 
    ConversationHandler
)

from automl_api.settings import load_config as load_config


config = {}

BASE_URL = ''

LOAD_DATASET = range(1)

(SELECT_PREDICT_MODEL,
SELECT_PREDICT_DATASET,
SELECT_PREDICT_PROBA,
PREDICT) = range(4)

(SELECT_DATASET,
SELECT_MODEL_TYPE,
SELECT_HYPERPARAMS,
SELECTED_MODELS,
FEATURE_SELECTION) = range(5)


def get_datasets():
    url = BASE_URL.format(endpoint='datasets')
    datasets = requests.get(url).json()

    datasets = datasets.get('datasets_files')

    return datasets


def get_models():
    url = BASE_URL.format(endpoint='models')
    models = requests.get(url).json()

    return models


def construct_models_msg():
    models = get_models()

    if not models:
        msg = 'Нет обученных моделей.\n\n'\
              'Обучить модель можно по команде /train\_model'

        return msg

    msgs = []

    for m in models:
        model = []
        
        _id = m['_id']['$oid']
        dataset = m['dataset'].replace('_', '\_')

        model.append(f'*id*: `{_id}`\n')
        model.append(f'*dataset*: `{dataset}`\n')

        m.pop('_id')
        m.pop('dataset')
        m.pop('filename')

        for k in m.keys():
            features = m[k]['selected_features']
            features = ' '.join(features)
            _k = k.replace('_', '\_')
            model.append(f'*{_k} features*: `{features}`\n')

        msg = ''.join(model).strip()

        msgs.append(msg)

    return msgs


def post_model_params(body):
    url = BASE_URL.format(endpoint='models')

    resp = requests.post(url, json=body)

    return resp.json()


def error(update, context):
    print(update, context.error)


def cancel(update, context):
    update.message.reply_text('Операция отменена')

    return ConversationHandler.END


def start(update, context):
    msg = '/start - показать это сообщение\n' \
          '/list\_models - список доступных обученных моделей\n' \
          '/list\_datasets - показать список загруженных датасетов\n' \
          '/add\_dataset - загрузить новый датасет\n' \
          '/train\_model - обучить модель\n' \
          '/predict - предсказать целевую переменныю'


    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=msg,
        parse_mode=telegram.ParseMode.MARKDOWN
        )


def list_models(update, context):

        msg = construct_models_msg()

        if isinstance(msg, list):
            for m in msg:
                context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text=m,
                    parse_mode=telegram.ParseMode.MARKDOWN,
                )

        else:
            context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=msg,
                parse_mode=telegram.ParseMode.MARKDOWN,
            )
            

def predict_data(update, context):

    models = get_models()
    models_msg = construct_models_msg() 

    if not models:
        context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=msg,
            parse_mode=telegram.ParseMode.MARKDOWN
        )

        return ConversationHandler.END
        

    models_ids = [m['_id']['$oid'] for m in models]

    reply_keyboard = [[m] for m in models_ids]

    reply_markup=None

    for m in models_msg:

        if m == models_msg[-1]:
            reply_markup=ReplyKeyboardMarkup(
                reply_keyboard, one_time_keyboard=True
            )
            

        context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=m,
            parse_mode=telegram.ParseMode.MARKDOWN,
            reply_markup=reply_markup
        )
    

    return SELECT_PREDICT_MODEL


def select_predict_model(update, context):
    context.user_data['predict_model'] = update.message.text

    datasets = get_datasets()

    context.user_data['available_datasets'] = datasets

    reply_keyboard = [[d] for d in datasets]

    msg = 'Выбери один из предложенных датасетов'

    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=msg,
        reply_markup=ReplyKeyboardMarkup(
            reply_keyboard, one_time_keyboard=True
        )
    )

    return SELECT_PREDICT_DATASET


def select_predict_dataset(update, context):
    context.user_data['selected_dataset'] = update.message.text

    msg = 'Использовать вывод в вероятностях?'
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=msg,
        parse_mode=telegram.ParseMode.MARKDOWN,
        reply_markup=ReplyKeyboardMarkup(
            [['Да'], ['Нет']], one_time_keyboard=True
        )
    )

    return SELECT_PREDICT_PROBA


def select_predict_proba(update, context):

    if update.message.text == 'Да':
        context.user_data['proba_predict'] = True
    else:
        context.user_data['proba_predict'] = False

    return predict_data_result(update, context)


def predict_data_result(update, context):
    dataset = context.user_data['selected_dataset']
    predict_model = context.user_data['predict_model']
    predict_proba = context.user_data['proba_predict']

    url = BASE_URL.format(endpoint='models/predict')

    body = {
        'dataset': dataset,
        'model_id': predict_model,
        'proba': predict_proba
    }

    resp = requests.post(url, json=body).json()

    if resp.get('status') == 'error':
        msg = f'*Ошибка* \n\nВывод в вероятностях доступен ' \
        'только для моделей классификации'

        context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=msg,
            parse_mode=telegram.ParseMode.MARKDOWN,
        )

        return SELECT_PREDICT_PROBA

    else:
        #f = TextIOWrapper(BytesIO(resp))

        import json
        f = json.dumps(resp).encode('utf-8')

        msg = 'Файл с результом предсказания ' \
              'сейчас будет загружен'

        context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=msg,
            parse_mode=telegram.ParseMode.MARKDOWN,
        )

        context.bot.send_document(
            chat_id=update.effective_chat.id,
            document=f,
            filename=f'{predict_model}.json'
        )

    return ConversationHandler.END


def train_model(update, context):
    msg = 'Чтобы добавить модель, я предложу ' \
          'поочередно ввести желаемые параметры'


    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=msg,
        parse_mode=telegram.ParseMode.MARKDOWN
    )

    datasets = get_datasets()

    context.user_data['available_datasets'] = datasets

    reply_keyboard = [[d] for d in datasets]

    msg = 'Выбери один из предложенных датасетов'

    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=msg,
        reply_markup=ReplyKeyboardMarkup(
            reply_keyboard, one_time_keyboard=True
        )
    )
    
    return SELECT_DATASET


def remove_model(update, context):
    import re

    user_msg = re.sub('\s+', ' ', update.message.text)

    try:
        _, model_id = user_msg.split(' ')
    except ValueError:
        msg = 'Необходимо после команды /remove\_model через пробел ' \
              'указать идентификатор модели. \n\n' \
              'Список доступных моделей можно увидеть набрав /list\_models'

        context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=msg,
            parse_mode=telegram.ParseMode.MARKDOWN
        )

        return
    
    url = BASE_URL.format(endpoint='models')

    body = {
        'model_id': model_id
    }

    resp = requests.delete(url, json=body).json()

    if resp['status'] == 'error':
        msg = resp["msg"].replace('_', '\_')
        msg = f'*Ошибка*\n\n{msg}'

    elif resp['status'] == 'success':
        msg = f'Модель `{model_id}` удалена'

    else:
        msg = '*Неизвестная ошибка*'

    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=msg,
        parse_mode=telegram.ParseMode.MARKDOWN
    )


def select_dataset(update, context):

    context.user_data['dataset'] = update.message.text

    model_types = config['automl']['available_models_classes'].keys()   

    reply_keyboard = [[mt] for mt in model_types]

    reply_keyboard.append(['По умолчанию'])

    msg = 'Выбери один из предложенных классов моделей'

    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=msg,
        reply_markup=ReplyKeyboardMarkup(
            reply_keyboard, one_time_keyboard=True
        )
    )

    return SELECT_MODEL_TYPE


def select_model_type(update, context):

    context.user_data['model_type'] = update.message.text

    if update.message.text == 'По умолчанию':
        context.user_data['model_type'] = ''

    models = config['automl']['available_models_classes'][context.user_data['model_type']]

    context.user_data['available_models'] = models
    
    msg = 'Опишите гиперпараметр для методов:' \
          f'*{" ,".join([m for m in models])}*\n' \
          'в следующем формате: \n' \
          '`lgbm: n_estimators=100 depth=5; catboost: n_estimators=100 depth=5;`'

    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=msg,
        parse_mode=telegram.ParseMode.MARKDOWN,
        reply_markup=ReplyKeyboardMarkup(
            [['По умолчанию']], one_time_keyboard=True
        )
    )
    
    return SELECT_HYPERPARAMS


def select_hyperparams(update, context):

    import re

    hyperparams = update.message.text

    if hyperparams == 'По умолчанию':
        context.user_data['hyperparams'] = ''

    else:

        hyperparams = hyperparams.strip(' ;')

        hyperparams = re.sub(r'[^\w\d=\s;:]', '', hyperparams)

        models = [m.strip() for m in hyperparams.split(';')]

        hyperparams = {}

        for m in models:
            model = m.split(':')[0].strip()

            if model not in context.user_data['available_models']:
                msg = '*Ошибка*\n' \
                    f'Модель `{model}` отстуствует в списке'

                context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text=msg,
                    parse_mode=telegram.ParseMode.MARKDOWN,
                    reply_markup=ReplyKeyboardMarkup(
                        [['По умолчанию']], one_time_keyboard=True
                    )
                )

                return SELECT_HYPERPARAMS

            params = m.split(':')[1].strip()

            params = [p.strip() for p in params.split(' ')]

            hyperparams[model] = {el.split('=')[0]:el.split('=')[1] for el in params}

        context.user_data['hyperparams'] = hyperparams

    msg = '*Выберите модели для обучения*\n\n' \
         f'Вы  можете выбрать одну из: `{", ".join(context.user_data["available_models"])}` ' \
          'или выбрать все'


    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=msg,
        parse_mode=telegram.ParseMode.MARKDOWN,
        reply_markup=ReplyKeyboardMarkup(
            [['all']], one_time_keyboard=True
        )
    )

    return SELECTED_MODELS


def select_selected_models(update, context):

    selected_models = update.message.text.strip(' ')

    if not selected_models == 'all':
        
        selected_models = [el.strip() for el in selected_models.split(',')]

        for m in selected_models:

            if m not in context.user_data['available_models']:
                msg = '*Ошибка*\n' \
                    f'Модель `{m}` отстуствует в списке'

                context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text=msg,
                    parse_mode=telegram.ParseMode.MARKDOWN,
                    reply_markup=ReplyKeyboardMarkup(
                        [['По умолчанию']], one_time_keyboard=True
                    )
                )

    context.user_data['selected_models'] = selected_models

    msg = 'Необходимо ли выполнять отбор признаков?'

    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=msg,
        parse_mode=telegram.ParseMode.MARKDOWN,
        reply_markup=ReplyKeyboardMarkup(
            [['Да'], ['Нет']], one_time_keyboard=True
        )
    )

    return FEATURE_SELECTION


def select_feature_selection(update, context):

    feature_selection = update.message.text

    if feature_selection == 'Да':
        feature_selection = True

    elif feature_selection == 'Нет':
        feature_selection = False

    context.user_data['feature_selection'] = feature_selection
    
    params = ['dataset', 'model_type', 'hyperparams', 'selected_models']

    body = {}

    for p in params:
        if context.user_data[p]:
            body[p] = context.user_data[p]

    resp = post_model_params(body)

    if resp['status'] == 'success':
        msg = 'Обучение модели добавлено в *очередь*. ' \
              'Когда модель обучится, ее можно будет найти ' \
              'в выводе команды /list\_models'

    else:
        msg = 'Ошибка, неверные параметры обучения'

    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=msg,
        parse_mode=telegram.ParseMode.MARKDOWN,
    )

    return ConversationHandler.END


def add_dataset(update, context):
    msg = 'Чтобы добавить датасет в коллекцию, ' \
          'отправь мне `.csv` файл. В датасете ' \
          'обязательно должно быть поле `target`'

    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=msg,
        parse_mode=telegram.ParseMode.MARKDOWN
    )

    return LOAD_DATASET
    

def load_dataset(update, context):
    

    f = context.bot.get_file(update.message.document)
    f = TextIOWrapper(BytesIO(f.download_as_bytearray()))
    
    filename = update.message.document.file_name

    url = BASE_URL.format(endpoint=f'datasets/{filename}')

    resp = requests.post(url, data=f).json()

    if resp.get('status', 'error') == 'success':
        msg = f'Добавлен датасет: `{resp["added_dataset"]}`'

    else:
        msg = f'*Ошибка*\n`{resp["msg"]}`'

    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=msg,
        parse_mode=telegram.ParseMode.MARKDOWN
    )

    return ConversationHandler.END


def list_datasets(update, context):

    datasets = get_datasets()

    msg = 'Доступные датасеты:\n'

    for ds in datasets:
        msg += f'*{ds}*\n'

    msg = msg.strip()

    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=msg,
        parse_mode=telegram.ParseMode.MARKDOWN
        )


def main(config_path):

    global config
    global BASE_URL

    config = load_config(config_path)

    api_host = config['automl_api']['api_host']
    api_port = config['automl_api']['api_port']

    BASE_URL = f'http://{api_host}:{api_port}'
    BASE_URL = BASE_URL + '/api/{endpoint}'

    api_key = config['telegram']['api_key']

    updater = Updater(api_key, use_context=True)

    dp = updater.dispatcher

    dp.add_handler(CommandHandler('start', start))
    dp.add_handler(CommandHandler('list_models', list_models))
    dp.add_handler(CommandHandler('list_datasets', list_datasets))
    dp.add_handler(CommandHandler('remove_model', remove_model))


    add_dataset_conv = ConversationHandler(
        entry_points=[CommandHandler('add_dataset', add_dataset)],
        states={
            LOAD_DATASET: [MessageHandler(Filters.document, load_dataset)]
        },
        fallbacks=[CommandHandler('cancel', cancel)]
    )

    predict_conv = ConversationHandler(
        entry_points=[CommandHandler('predict', predict_data)],
        states={
            SELECT_PREDICT_MODEL: [
                MessageHandler(
                    Filters.text & ~Filters.command,
                    select_predict_model
                )
            ],
            SELECT_PREDICT_DATASET: [
                MessageHandler(
                    Filters.text & ~Filters.command,
                    select_predict_dataset
                )
            ],
            SELECT_PREDICT_PROBA: [
                MessageHandler(
                    Filters.text & ~Filters.command,
                    select_predict_proba
                )
            ],
            PREDICT: [
                MessageHandler(
                    Filters.text & ~Filters.command,
                    predict_data_result
                )
            ]
        },
        fallbacks=[CommandHandler('cancel', cancel)]
    )

    train_model_conv = ConversationHandler(
        entry_points=[CommandHandler('train_model', train_model)],
        states={
            SELECT_DATASET: [
                MessageHandler(Filters.text & ~Filters.command,
                select_dataset)
            ],
            SELECT_MODEL_TYPE: [
                  MessageHandler(Filters.text & ~Filters.command,
                select_model_type)  
            ],
            SELECT_HYPERPARAMS: [
                MessageHandler(Filters.text & ~Filters.command,
                select_hyperparams)
            ],
            SELECTED_MODELS: [
                MessageHandler(Filters.text & ~Filters.command,
                select_selected_models)
            ],
            FEATURE_SELECTION: [
                MessageHandler(Filters.regex('^(Да|Нет)$'),
                select_feature_selection)
            ]
        },
        fallbacks=[CommandHandler('cancel', cancel)]
    )

    dp.add_handler(add_dataset_conv)
    dp.add_handler(train_model_conv)
    dp.add_handler(predict_conv)

    dp.add_error_handler(error)

    updater.start_polling()

    updater.idle()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config')
    
    args = parser.parse_args()

    if args.config:
        main(args.config)

    else:
        parser.print_help()
    
